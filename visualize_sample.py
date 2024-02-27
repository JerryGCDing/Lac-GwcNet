import torch.backends.cudnn as cudnn
import torch.cuda
import numpy as np
import os

from test_ds import calc_voxel_grid
from dataloader.ds_loader import VoxelDSDatasetCalib
from networks.stackhourglass import PSMNet

SAVE_DIR = './visualization/demo/'

cudnn.benchmark = True
sample_dataset = VoxelDSDatasetCalib('/work/vig/Datasets/DrivingStereo',
                                   './filenames/DS_test_gt_calib.txt',
                                   False,
                                   [-8, 10, -3, 3, 0, 30],
                                   [3, 1.5, 0.75, 0.375])


def main():
    affinity_settings = {}
    affinity_settings['win_w'] = 3
    affinity_settings['win_h'] = 3
    affinity_settings['dilation'] = [1, 2, 4, 8]
    model = PSMNet(maxdisp=192, struct_fea_c=4, fuse_mode='separate',
                   affinity_settings=affinity_settings, udc=True, refine='csr')
    if torch.cuda.is_available():
        model.cuda()

    state_dict = torch.load('/scratch/ding.tian/logs_ddp/Lac-Gwc_ft/checkpoint_3.ckpt')['net']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    for idx in [1000, 1500, 2000]:
        file_id = os.path.split(sample_dataset[idx]['left_filename'])[-1].split('.')[0]

        imgL = sample_dataset[idx]['left'][None, ...]
        imgR = sample_dataset[idx]['right'][None, ...]
        colored_cloud_gt = torch.from_numpy(
            np.frombuffer(sample_dataset[idx]['point_cloud'], dtype=np.float32).reshape(-1, 6))
        '''
        voxel_gt = sample_dataset[idx]['voxel_grid']
        np.savez(os.path.join(SAVE_DIR, f'gt_{file_id}.npz'), colored_pc_gt=colored_cloud_gt,
                 level_0=voxel_gt[0].numpy(), level_1=voxel_gt[1].numpy(), level_2=voxel_gt[2].numpy(),
                 level_3=voxel_gt[3].numpy())
        '''

        if torch.cuda.is_available():
            imgL = imgL.cuda()
            imgR = imgR.cuda()

        with torch.no_grad():
            disp_est = model(imgL, imgR)[-1].squeeze().cpu().numpy()
            assert len(disp_est.shape) == 2
            disp_est[disp_est <= 0] -= 1.

        depth_est = sample_dataset.f_u * 0.54 / disp_est
        cloud_est = sample_dataset.calc_cloud(depth_est)
        filtered_cloud_est = sample_dataset.filter_cloud(cloud_est)
        voxel_est, _ = calc_voxel_grid(filtered_cloud_est, (48, 16, 80), .375)

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        np.savez(os.path.join(SAVE_DIR, f'{file_id}.npz'), colored_pc_gt=colored_cloud_gt, level_3=voxel_est)


if __name__ == '__main__':
    main()
