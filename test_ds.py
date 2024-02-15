import coloredlogs
import logging
import torch.backends.cudnn as cudnn
import torch.cuda
from pytorch3d.loss import chamfer_distance
from thop import profile, clever_format
from tqdm import tqdm
import numpy as np
import time
import argparse
import os

from dataloader.ds_loader import VoxelDSDatasetCalib
from networks.stackhourglass import PSMNet


parser = argparse.ArgumentParser(description='LaC')
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data_path', type=str, default='/work/vig/Datasets/DrivingStereo/')
parser.add_argument('--load_path', type=str,
                    default='/scratch/ding.tian/logs_ddp/Lac-Gwc_ft/checkpoint_3.ckpt')
parser.add_argument('--max_disp', type=int, default=192)
parser.add_argument('--lsp_width', type=int, default=3)
parser.add_argument('--lsp_height', type=int, default=3)
parser.add_argument('--lsp_dilation', type=list, default=[1, 2, 4, 8])
parser.add_argument('--lsp_mode', type=str, default='separate')
parser.add_argument('--lsp_channel', type=int, default=4)
parser.add_argument('--no_udc', action='store_true', default=False)
parser.add_argument('--refine', type=str, default='csr')
args = parser.parse_args()

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

BATCH_SIZE = 1
MAXDISP = 192

cudnn.benchmark = True
if not args.no_cuda:
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)


def calc_IoU(pred, gt):
    intersect = pred * gt
    total = pred + gt
    union = total - intersect

    return (intersect.sum() + 1.0) / (union.sum() + 1.0)


def eval_metric(voxel_ests, voxel_gt, metric_func, *args, depth_range=None):
    """

    @param voxel_ests:
    @param voxel_gt:
    @param metric_func:
    @param depth_range:
    @return: Dict{%{depth_range[0]}: [lv0, lv1, lv2, ...], %{depth_range[1]}: [...], ...}
    """
    if depth_range is None:
        depth_range = [1.]

    out_dict = {}
    for r in depth_range:
        out_dict[str(r)] = []

    if isinstance(voxel_ests, torch.Tensor):
        est_shape = voxel_ests.shape
        for idx in range(len(voxel_gt) - 1, -1, -1):
            gt_shape = voxel_gt[idx].shape
            if est_shape[1] == gt_shape[1] and est_shape[2] == gt_shape[2] and est_shape[3] == gt_shape[3]:
                for depth_r in depth_range:
                    z = int(est_shape[-1] * depth_r)
                    if len(args) == 0:
                        metric = metric_func(voxel_ests[..., :z], voxel_gt[idx][..., :z])
                    else:
                        metric = metric_func(voxel_ests[..., :z], voxel_gt[idx][..., :z], args[0][idx])
                    out_dict[str(depth_r)].append(metric)

    elif isinstance(voxel_ests[0], torch.Tensor):
        for idx, voxel_est in enumerate(voxel_ests):
            for depth_r in depth_range:
                z = int(voxel_est.shape[-1] * depth_r)
                if len(args) == 0:
                    metric = metric_func(voxel_est[..., :z], voxel_gt[idx][..., :z])
                else:
                    metric = metric_func(voxel_est[..., :z], voxel_gt[idx][..., :z], args[0][idx])
                out_dict[str(depth_r)].append(metric)

    else:
        raise NotImplementedError

    return out_dict


test_dataset = VoxelDSDatasetCalib(args.data_path,
                                   './filenames/DS_test_gt_calib.txt',
                                   False,
                                   [-8, 10, -3, 3, 0, 30],
                                   [3, 1.5, 0.75, 0.375])
affinity_settings = {}
affinity_settings['win_w'] = args.lsp_width
affinity_settings['win_h'] = args.lsp_width
affinity_settings['dilation'] = args.lsp_dilation
udc = not args.no_udc

model = PSMNet(maxdisp=args.max_disp, struct_fea_c=args.lsp_channel, fuse_mode=args.lsp_mode,
               affinity_settings=affinity_settings, udc=udc, refine=args.refine)


def calc_voxel_grid(filtered_cloud, grid_size, voxel_size):
    # quantized point values, here you will loose precision
    xyz_q = np.floor(np.array(filtered_cloud / voxel_size)).astype(int)
    # Empty voxel grid
    vox_grid = np.zeros(grid_size)
    offsets = np.array([int(9 / voxel_size), int(3 / voxel_size), 0])
    xyz_offset_q = np.clip(xyz_q + offsets, [0, 0, 0], np.array(grid_size) - 1)
    # Setting all voxels containitn a points equal to 1
    vox_grid[xyz_offset_q[:, 0], xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1

    # get back indexes of populated voxels
    xyz_v = np.asarray(np.where(vox_grid == 1))
    cloud_np = np.asarray([(pt - offsets) * voxel_size for pt in xyz_v.T])
    return torch.from_numpy(vox_grid), cloud_np


def eval_model():
    if torch.cuda.is_available():
        model.cuda()

    state_dict = torch.load(args.load_path)['net']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    iou_dict = MetricDict()
    cd_dict = MetricDict()
    infer_time = []
    for batch_idx, sample in enumerate(tqdm(test_dataset)):
        imgL = sample['left'][None, ...]
        imgR = sample['right'][None, ...]
        voxel_gt = sample['voxel_grid'][-1]

        if torch.cuda.is_available():
            imgL = imgL.cuda()
            imgR = imgR.cuda()

        start = time.time()
        with torch.no_grad():
            disp_est = model(imgL, imgR, torch.zeros_like(imgL).cuda()).squeeze().cpu().numpy()
            assert len(disp_est.shape) == 2
            disp_est[disp_est <= 0] -= 1.

        depth_est = test_dataset.f_u * 0.54 / disp_est
        cloud_est = test_dataset.calc_cloud(depth_est)
        filtered_cloud_est = test_dataset.filter_cloud(cloud_est)
        voxel_est, _ = calc_voxel_grid(filtered_cloud_est, (48, 16, 80), .375)
        infer_time.append(time.time() - start)

        iou_dict.append(eval_metric([voxel_est], [voxel_gt], calc_IoU, depth_range=[.5, 1.]))
        cd_dict.append(eval_metric([voxel_est], [voxel_gt], eval_cd, [0.375], depth_range=[.5, 1.]))

    iou_mean = iou_dict.mean()
    cd_mean = cd_dict.mean()

    for k in iou_mean.keys():
        msg = f'Depth - {k}: IoU = {str(iou_mean[k].tolist())}; CD = {str(cd_mean[k].tolist())}'
        logger.info(msg)
    avg_infer = np.mean(np.array(infer_time))
    logger.info(f'Avg_infer = {avg_infer}; FPS = {1 / avg_infer}')


def eval_cd(pred, gt, scale):
    pred_coord = torch.nonzero((pred.squeeze(0) >= 0.5).int()) * float(scale)
    gt_coord = torch.nonzero((gt.squeeze(0) == 1).int()) * float(scale)

    return chamfer_distance(pred_coord[None, ...], gt_coord[None, ...])[0]


class MetricDict:
    def __init__(self):
        self._data = {}

    def append(self, in_dict):
        for k, v, in in_dict.items():
            if k not in self._data:
                self._data[k] = [v]
            else:
                self._data[k].append(v)

    def mean(self):
        out_dict = {}
        for k, v in self._data.items():
            v_t = torch.asarray(v)
            out_dict[k] = torch.mean(v_t, dim=0)

        return out_dict

    def __getattr__(self, item):
        return getattr(self._data, item)()

    def __getitem__(self, item):
        return self._data[item]


def eval_ops():
    if torch.cuda.is_available():
        model.cuda()

    sample = test_dataset[0]
    imgL, imgR, voxel_gt = sample['left'][None, ...], sample['right'][None, ...], sample['voxel_grid']
    if torch.cuda.is_available():
        imgL = imgL.cuda()
        imgR = imgR.cuda()
    with torch.no_grad():
        macs, params = clever_format(profile(model, inputs=(imgL, imgR, torch.zeros_like(imgL).cuda())), '%.3f')

    print(f'MACS: {macs}, PARAMS: {params}')


if __name__ == '__main__':
    eval_model()
    # eval_ops()
