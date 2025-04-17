import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision
import warnings
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.spynet_arch import SpyNet
from basicsr.ops.dcn import ModulatedDeformConvPack
from basicsr.archs.attn_util import DualChannelBlock
from .arch_util import flow_warp, ConvResidualBlocks, EventFrameAlign


@ARCH_REGISTRY.register()
class EvDeblurVSR(nn.Module):

    """Event-Enhanced Blurry Video Super-Resolution (AAAI 2025)

       Note that: this class is for 4x blurry VSR

    Args:
        mid_channels (int): Channel number of the intermediate features. Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch. Default: 7.
        voxel_bins (int): Number of voxel bins. Default: 5.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 voxel_bins=5,
                 max_residue_magnitude=10,
                 spynet_path=None):

        super().__init__()
        self.mid_channels = mid_channels
        self.voxel_bins = voxel_bins

        # optical flow
        self.spynet = SpyNet(spynet_path)

        # feature extraction module
        self.feat_extract_img = ConvResidualBlocks(3, mid_channels, 5)
        self.feat_extract_eve = ConvResidualBlocks(voxel_bins, mid_channels, 5)

        # Reciprocal Feature Deblurring
        self.RFD = DualChannelBlock(dim=mid_channels)

        # Hybrid Deformable Alignment
        self.event_align1 = EventFrameAlign(mid_channels,voxel_bins)
        self.event_align2 = EventFrameAlign(mid_channels,voxel_bins*2)
 
        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment_Event(
                    in_channels = 2 * mid_channels,
                    out_channels = mid_channels,
                    kernel_size = 3,
                    padding=1,
                    deformable_groups=16,
                    max_residue_magnitude=max_residue_magnitude,
                    voxel_bins=voxel_bins)
            self.backbone[module] = ConvResidualBlocks((2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ConvResidualBlocks(5 * mid_channels, mid_channels, 5)

        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn('Deformable alignment module is not added. '
                          'Probably your CUDA is not configured correctly. DCN can only '
                          'be used with CUDA enabled. Alignment is skipped now.')


    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the flows used for forward-time propagation \
                (current to previous). 'flows_backward' corresponds to the flows used for backward-time \
                propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def propagate(self, feats, flows, voxels, module_name):
        """Propagate the latent features throughout the sequence with event

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            voxels (tensor): Propagation voxels with shape (n, t - 1, Bins, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated \
                features. Each key in the dictionary corresponds to a \
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial_i'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current_i = feats['spatial_i'][mapping_idx[idx]]
            feat_current_e = feats['spatial_e'][mapping_idx[idx]]

            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                voxel_prop_n1 = voxels[:, flow_idx[i], :, :, :]
                feat_voxel_n1 = self.event_align1(feat_prop, voxel_prop_n1)

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                feat_voxel_n2 = torch.zeros_like(feat_voxel_n1)
                voxel_prop_n2 = torch.zeros(n, self.voxel_bins*2, h, w).cuda()

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                    voxel_prop_n2 = voxels[:, flow_idx[i - 1], :, :, :]
                    voxel_prop_n2 = torch.cat([voxel_prop_n2, voxel_prop_n1], dim=1) # [B, bins*2, H, W]
                    feat_voxel_n2 = self.event_align2(feat_n2, voxel_prop_n2)

                # hybrid deformable alignment, concat elements to construct condition pool
                cond = torch.cat([cond_n1, feat_current_i, cond_n2, feat_current_e, feat_voxel_n1, feat_voxel_n2, voxel_prop_n1, voxel_prop_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current_i] + [feats[k][idx] for k in feats if k not in ['spatial_i', 'spatial_e', module_name]] + [feat_prop]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []
        num_outputs = len(feats['spatial_i'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k not in ['spatial_i', 'spatial_e']]
            hr.insert(0, feats['spatial_i'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            hr += self.img_upsample(lqs[:, i, :, :, :])

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs, vExpos, vFwds, vBwds):
        """Forward function for EvDeblurVSR

        Args:
            lqs (tensor): Input low quality (LQ) sequence with shape (n, t, c, h, w).
            vExpos (tensor): Intra-frame voxels with shape (n, t, bins, h, w).
            vFwds (tensor): Inter-frame forward voxels with shape (n, t-1, bins, h, w).
            vBwds (tensor): Inter-frame backward voxels with shape (n, t-1, bins, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c1, h, w = lqs.size()
        _, _, c2, _, _ = vExpos.size()

        lqs_downsample = lqs.clone()


        feats = {}
        # compute spatial features, shallow feature extractor
        feats_i = self.feat_extract_img(lqs.view(-1, c1, h, w))
        feats_e = self.feat_extract_eve(vExpos.view(-1, c2, h, w))

        # reciprocal feature deblurring
        feats_i, feats_e = self.RFD(feats_i, feats_e) # [B*T, dim, H, W]

        h, w = feats_i.shape[2:]
        feats_i = feats_i.view(n, t, -1, h, w)
        feats_e = feats_e.view(n, t, -1, h, w)

        feats['spatial_i'] = [feats_i[:, i, :, :, :] for i in range(0, t)]
        feats['spatial_e'] = [feats_e[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                    voxels = vBwds
                elif flows_forward is not None:
                    flows = flows_forward
                    voxels = vFwds
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, voxels, module)

        return self.upsample(lqs, feats)


class SecondOrderDeformableAlignment_Event(ModulatedDeformConvPack):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.voxel_bins = kwargs.pop('voxel_bins', 5)

        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(6 * self.out_channels + 4 + self.voxel_bins * 3, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)