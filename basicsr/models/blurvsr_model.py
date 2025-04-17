import torch
from collections import Counter
from collections import OrderedDict
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.data.transforms import mod_crop
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class EvBlurVSRModel(VideoBaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.scale = opt['scale']
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('edge_opt'):
            self.cri_edge = build_loss(train_opt['edge_opt']).to(self.device)
        else:
            self.cri_edge = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_edge is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.voxels_f = data['voxels_f'].to(self.device)
        self.voxels_b = data['voxels_b'].to(self.device)
        self.voxels_e = data['voxels_e'].to(self.device)
        if 'key' in data:
            self.key = data['key']
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'voxels_e_hr' in data:
            self.voxels_e_hr = data['voxels_e_hr'].to(self.device)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.voxels_e, self.voxels_f, self.voxels_b)

        # define loss
        l_total = 0
        loss_dict = OrderedDict()
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            # check loss
            if torch.isnan(l_pix).any():
                print("self.key: ", self.key)
                print("torch.isnan(self.output).any(): ", torch.isnan(self.output).any())
                raise ValueError(f"l_pix is nan. Please check.")
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_edge:
            # calculate edge weight
            edge_weight = torch.abs(self.voxels_e_hr).sum(dim=2, keepdim=True) # [B, T, 1, H, W]
            # edge_weight = edge_weight.expand(-1, -1, 3, -1, -1)  # [B, T, 3, H, W]
            min_val = edge_weight.min()
            max_val = edge_weight.max()
            range_val = max_val - min_val

            if range_val > 0:
                edge_weight = (edge_weight - min_val) / range_val
            else:
                edge_weight = torch.zeros_like(edge_weight)

            l_edge = self.cri_edge(self.output, self.gt, weight=edge_weight)

            # check loss
            if torch.isnan(l_edge).any():
                print("self.key: ", self.key)
                print("torch.isnan(self.output).any(): ", torch.isnan(self.output).any())
                raise ValueError(f"l_edge is nan. Please check.")
            l_total += l_edge
            loss_dict['l_edge'] = l_edge

        # loss_dict['l_total'] = l_total
        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                num_frame_each_folder = Counter(dataset.data_info['folder'])
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        metric_data = dict()
        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
        # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
        # (To avoid wait-dead)
        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']

            # compute outputs
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            val_data['voxels_f'].unsqueeze_(0)
            val_data['voxels_b'].unsqueeze_(0)
            val_data['voxels_e'].unsqueeze_(0)

            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)
            val_data['voxels_f'].squeeze_(0)
            val_data['voxels_b'].squeeze_(0)
            val_data['voxels_e'].squeeze_(0)

            self.test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.voxels_f
            del self.voxels_b
            del self.voxels_e
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            # evaluate
            if i < num_folders:
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img([result])  # uint8, bgr
                    metric_data['img'] = result_img
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr
                        gt_img = mod_crop(gt_img, self.scale)
                        metric_data['img2'] = gt_img

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            result = calculate_metric(metric_data, opt_)
                            self.metric_results[folder][idx, metric_idx] += result

                    if save_img:
                        if self.opt['is_train']:
                            raise NotImplementedError('saving image is not supported during training.')
                        else:
                            temp_psnr = self.metric_results[folder][idx, 0]
                            img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                folder, f"{idx:06d}_{temp_psnr:.4f}_{self.opt['name']}.png")
                        imwrite(result_img, img_path)

                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def test(self):
        self.net_g.eval()

        self.max_seq_len = self.opt['val'].get('max_seq_len', None)

        with torch.no_grad():
            if self.max_seq_len:
                self.output = []
                for i in range(0, self.lq.size(1), self.max_seq_len):
                    imgs = self.lq[:, i:i+self.max_seq_len, :, :, :]
                    ve = self.voxels_e[:, i:i+self.max_seq_len, :, :, :]
                    vf = self.voxels_f[:, i:i+self.max_seq_len-1, :, :, :]
                    vb = self.voxels_b[:, i:i+self.max_seq_len-1, :, :, :]
                    self.output.append(self.net_g(imgs, ve, vf, vb))
                self.output = torch.cat(self.output, dim=1)
            else:
                self.output = self.net_g(self.lq, self.voxels_e, self.voxels_f, self.voxels_b)

        self.net_g.train()

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):

        metric_results_avg = {
            folder: torch.mean(tensor, dim=0).cpu()
            for (folder, tensor) in self.metric_results.items()
        }

        cnt_folder = {folder: tensor.size(0) for (folder, tensor) in self.metric_results.items()}
        total_avg_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        for folder, tensor in metric_results_avg.items():
            for metric_idx, metric in enumerate(total_avg_results.keys()):
                total_avg_results[metric] += tensor[metric_idx].item() * cnt_folder[folder]

        total_samples_length = 0
        for folder, length in cnt_folder.items():
            total_samples_length += length

        # average among folders
        for metric in total_avg_results.keys():
            total_avg_results[metric] /= total_samples_length
            # update the best metric result
            self._update_best_metric_result(dataset_name, metric, total_avg_results[metric], current_iter)

        # ------------------------------------------ log the metric ------------------------------------------ #
        log_str = f'Validation {dataset_name}\n'

        # sort the metric_results_avg by key
        metric_results_avg = {key: metric_results_avg[key] for key in sorted(metric_results_avg)}
        for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
            log_str += f'\t # {metric}: {value:.4f}'
            for folder, tensor in metric_results_avg.items():
                log_str += f'\n\t # {folder}: {tensor[metric_idx].item():.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\n\t    Best: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
                tb_logger.add_scalar(f'metrics/Best_{metric}', float(f'{self.best_metric_results[dataset_name][metric]["val"]:.4f}'), current_iter)
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
                for folder, tensor in metric_results_avg.items():
                    tb_logger.add_scalar(f'metrics/{metric}/{folder}', tensor[metric_idx].item(), current_iter)
