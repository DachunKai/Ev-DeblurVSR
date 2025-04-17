import numpy as np
import torch
from os import path as osp
from torch.utils import data as data

from basicsr.utils import get_root_logger, FileClient
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import mod_crop
from basicsr.utils.img_util import img2tensor

@DATASET_REGISTRY.register()
class EvBlurVSRTestDataset(data.Dataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.need_expo = opt.get('need_expo', True)
        self.data_info = {'folder': []}
        self.scale = opt['scale']
        self.name = opt['name']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if self.io_backend_opt['type'] == 'hdf5':
            self.io_backend_opt['h5_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['LR', 'HR']
        else:
            raise ValueError(f"We don't realize {self.io_backend_opt['type']} backend")

        logger = get_root_logger()
        logger.info(f'Generate data info for BlurAlignTestDataset - {opt["name"]}')

        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split(' ')[0])
                    clips_num.append(line.split(' ')[1])
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt, self.voxels_f, self.voxels_b, self.voxels_e = {}, {}, {}, {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip(clips, clips_num):
            self.io_backend_opt['h5_clip'] = clip
            clip_name = osp.splitext(clip)[0]
            self.file_client = FileClient(self.io_backend_opt['type'], **self.io_backend_opt)

            img_lqs, img_gts, voxels_f, voxels_b, voxels_Expo = self.file_client.get(list(range(int(num))))

            # mod_crop gt image for scale
            img_gts = [mod_crop(img, self.scale) for img in img_gts]
            self.imgs_lq[clip_name] = torch.stack(img2tensor(img_lqs), dim=0)
            self.imgs_gt[clip_name] = torch.stack(img2tensor(img_gts), dim=0)
            self.voxels_f[clip_name] = torch.from_numpy(np.stack(voxels_f, axis=0))
            self.voxels_b[clip_name] = torch.from_numpy(np.stack(voxels_b, axis=0))
            self.voxels_e[clip_name] = torch.from_numpy(np.stack(voxels_Expo, axis=0))
            self.folders.append(clip_name)
            self.lq_paths.append(osp.join(f'{self.name}', clip_name))
            self.data_info['folder'].extend([clip_name] * int(num))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.lq_paths[index]

        img_lq = self.imgs_lq[folder]
        img_gt = self.imgs_gt[folder]
        voxels_f = self.voxels_f[folder]
        voxels_b = self.voxels_b[folder]
        voxels_e = self.voxels_e[folder]

        if self.need_expo:
            return {
                'lq': img_lq,
                'gt': img_gt,
                'voxels_f': voxels_f,
                'voxels_b': voxels_b,
                'voxels_e': voxels_e,
                'folder': folder,
                'lq_path': lq_path
            }
        else:
            return {
                'lq': img_lq,
                'gt': img_gt,
                'voxels_f': voxels_f,
                'voxels_b': voxels_b,
                'folder': folder,
                'lq_path': lq_path
            }

    def __len__(self):
        return len(self.folders)