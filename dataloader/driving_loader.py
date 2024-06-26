import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import warnings

from .data_io import get_transform, read_all_lines, readPFM
from .wrappers import Camera, Pose
from .voxel_dataset import VoxelDataset


class DrivingDataset(Dataset):
    def __init__(self, datapath, list_filename, training, transform=True):
        self.datapath = datapath
        self.left_filenames = None
        self.right_filenames = None
        self.disp_filenames = None
        self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None
        self.transform = transform

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        self.left_filenames = []
        self.right_filenames = []
        self.disp_filenames = []

        self.focal_lengths = []
        for x in splits:
            self.left_filenames.append(x[0])
            self.right_filenames.append(x[1])
            self.disp_filenames.append(x[2])

            self.focal_lengths.append(x[-2])

    def load_image(self, filename, *, store_id=False):
        if store_id:
            self.file_id = int(filename.split('/')[-1].split('.')[0])
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, _ = readPFM(filename)

        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(
            self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(
            self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(
            self.datapath, self.disp_filenames[index]))

        w, h = left_img.size
        crop_w, crop_h = 960, 540

        processed = get_transform()

        if self.transform:
            if w < crop_w:
                left_img = processed(left_img).numpy()
                right_img = processed(right_img).numpy()

                left_img = np.lib.pad(
                    left_img, ((0, 0), (0, 0), (0, crop_w-w)), mode='constant', constant_values=0)
                right_img = np.lib.pad(
                    right_img, ((0, 0), (0, 0), (0, crop_w-w)), mode='constant', constant_values=0)
                disparity = np.lib.pad(
                    disparity, ((0, 0), (0, crop_w-w)), mode='constant', constant_values=0)

                left_img = torch.Tensor(left_img)
                right_img = torch.Tensor(right_img)
            else:
                left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
                right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
                disparity = disparity[h - crop_h:h, w - crop_w: w]

                left_img = processed(left_img)
                right_img = processed(right_img)
        else:
            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            left_img = np.asarray(left_img)
            right_img = np.asarray(right_img)

        return {"left": left_img,
                "right": right_img,
                "disparity": disparity,
                "left_filename": self.left_filenames[index]}


class VoxelDrivingDataset(VoxelDataset):
    def __init__(self, datapath, list_filename, training, roi_scale, voxel_sizes, transform=True, *, filter_ground=True,
                 color_jitter=False, occupied_gates=(None, None, 10, 5)):
        super().__init__(datapath, roi_scale, voxel_sizes, transform, filter_ground=filter_ground,
                         color_jitter=color_jitter, occupied_gates=occupied_gates)
        self.left_filenames = None
        self.right_filenames = None
        self.disp_filenames = None
        self.gt_voxel_filenames = None
        self.focal_lengths = None
        self.calib_filepaths = None
        self.load_path(list_filename)
        if training:
            assert self.disp_filenames is not None

        self.file_id = None
        self.baseline = 1.0
        self.ground_y = 0.9
        self.img_res = (960, 540)

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        self.left_filenames = []
        self.right_filenames = []
        self.disp_filenames = []

        self.focal_lengths = []
        self.calib_filepaths = []
        for x in splits:
            self.left_filenames.append(x[0])
            self.right_filenames.append(x[1])
            self.disp_filenames.append(x[2])

            self.focal_lengths.append(x[-2])
            self.calib_filepaths.append(x[-1])

        # stored gt available
        if len(splits[0]) > 5:
            self.stored_gt = True
            self.gt_voxel_filenames = [x[3] for x in splits]

    def load_image(self, filename, *, store_id=False):
        if store_id:
            self.file_id = int(filename.split('/')[-1].split('.')[0])
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, _ = readPFM(filename)

        return data

    def load_calib(self, filepath):
        line_idx = 4 * (self.file_id - 1)
        extrinsic_file = os.path.join(filepath, 'camera_data.txt')

        with open(extrinsic_file, 'r') as f:
            lines = f.readlines()
            assert int(lines[line_idx].split()[-1]) == self.file_id
            line_1 = lines[line_idx + 1]
            line_3 = lines[line_idx + 2]

        split_1 = line_1.split()
        split_3 = line_3.split()
        assert split_1[0] == 'L' and split_3[0] == 'R'
        extrinsic_1 = np.array(list(map(float, split_1[1:]))).reshape(4, 4)
        extrinsic_3 = np.array(list(map(float, split_3[1:]))).reshape(4, 4)

        extrinsic_norm = np.linalg.inv(extrinsic_1)
        extrinsic_01 = extrinsic_1 @ extrinsic_norm
        extrinsic_03 = extrinsic_3 @ extrinsic_norm

        T_world_cam_01 = np.concatenate(
            [extrinsic_01[:3, :3].flatten(), [0., 0., 0.]], axis=-1)
        T_world_cam_03 = np.concatenate(
            [extrinsic_03[:3, :3].flatten(), extrinsic_03[:3, 3]], axis=-1)

        cam_01 = np.array([self.f_u, self.f_v, self.c_u, self.c_v])
        cam_03 = np.array([self.f_u, self.f_v, self.c_u, self.c_v])

        T_world_cam_101 = T_world_cam_01.astype(np.float32)
        cam_101 = cam_01.astype(np.float32)
        T_world_cam_103 = T_world_cam_03.astype(np.float32)
        cam_103 = cam_03.astype(np.float32)

        self.lidar_extrinsic = Pose(T_world_cam_101)
        return T_world_cam_101, cam_101, T_world_cam_103, cam_103

    def calc_cloud(self, disp):
        depth_gt = self.f_u * self.baseline / disp
        mask = (depth_gt > 0).reshape(-1)

        rows, cols = depth_gt.shape
        x, y = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
        points = np.stack([x, y, depth_gt], axis=-1).reshape(-1, 3)
        points = points[mask]
        cloud = self.project_image_to_velo(points)

        return cloud

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]), store_id=True)
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        self.c_u = 479.5
        self.c_v = 269.5
        if self.focal_lengths[index] == '15mm':
            self.f_u = 450.0
            self.f_v = 450.0
        elif self.focal_lengths[index] == '35mm':
            self.f_u = 1050.0
            self.f_v = 1050.0
        else:
            raise RuntimeError('Unrecognizable camera focal length')
        T_world_cam_101, cam_101, T_world_cam_103, cam_103 = self.load_calib(
            os.path.join(self.datapath, self.calib_filepaths[index]))
        disp_gt = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        # numpy to tensor
        T_world_cam_101 = torch.from_numpy(T_world_cam_101)
        T_world_cam_103 = torch.from_numpy(T_world_cam_103)

        w, h = left_img.size
        crop_w, crop_h = self.img_res

        processed = get_transform(self.color_jitter)
        left_top = [0, 0]

        if self.transform:
            if w < crop_w:
                left_img = processed(left_img).numpy()
                right_img = processed(right_img).numpy()

                w_pad = crop_w - w
                left_img = np.lib.pad(
                    left_img, ((0, 0), (0, 0), (0, w_pad)), mode='constant', constant_values=0)
                right_img = np.lib.pad(
                    right_img, ((0, 0), (0, 0), (0, w_pad)), mode='constant', constant_values=0)
                disp_gt = np.lib.pad(
                    disp_gt, ((0, 0), (0, w_pad)), mode='constant', constant_values=0)

                left_img = torch.Tensor(left_img)
                right_img = torch.Tensor(right_img)
            else:
                w_crop = w - crop_w
                h_crop = h - crop_h
                left_img = left_img.crop((w_crop, h_crop, w, h))
                right_img = right_img.crop((w_crop, h_crop, w, h))
                disp_gt = disp_gt[h_crop: h, w_crop: w]

                left_img = processed(left_img)
                right_img = processed(right_img)
                left_top = [w_crop, h_crop]
        else:
            w_crop = w - crop_w
            h_crop = h - crop_h
            left_img = left_img.crop((w_crop, h_crop, w, h))
            right_img = right_img.crop((w_crop, h_crop, w, h))
            left_img = np.asarray(left_img)
            right_img = np.asarray(right_img)
            left_top = [w_crop, h_crop]

        left_top = np.repeat(np.array([left_top]), repeats=2, axis=0)

        all_vox_grid_gt = []
        cloud_gt = self.calc_cloud(disp_gt)
        filtered_cloud_gt = self.filter_cloud(cloud_gt)

        if self.stored_gt:
            all_vox_grid_gt = self.load_gt(os.path.join(self.datapath, self.gt_voxel_filenames[index]))
            valid_gt, _ = self.calc_voxel_grid(filtered_cloud_gt, 0)
            if not torch.allclose(all_vox_grid_gt[0], torch.from_numpy(valid_gt)):
                warnings.warn(
                    f'Stored label inconsistent.\n Loaded gt: \n {all_vox_grid_gt[0]} \n Validate gt: \n '
                    f'{valid_gt}')
        else:
            parent_grid = None
            try:
                for level in range(len(self.grid_sizes)):
                    vox_grid_gt, cloud_np_gt = self.calc_voxel_grid(
                        filtered_cloud_gt, level=level, parent_grid=parent_grid)
                    vox_grid_gt = torch.from_numpy(vox_grid_gt)

                    parent_grid = vox_grid_gt
                    all_vox_grid_gt.append(vox_grid_gt)
            except Exception as e:
                raise RuntimeError('Error in calculating voxel grids from point cloud')

        imc, imh, imw = left_img.shape
        cam_101 = np.concatenate(([imw, imh], cam_101)).astype(np.float32)
        cam_103 = np.concatenate(([imw, imh], cam_103)).astype(np.float32)

        return {'left': left_img,
                'right': right_img,
                'T_world_cam_101': T_world_cam_101,
                'cam_101': cam_101,
                'T_world_cam_103': T_world_cam_103,
                'cam_103': cam_103,
                'voxel_grid': all_vox_grid_gt,
                'point_cloud': filtered_cloud_gt.astype(np.float32).tobytes(),
                'left_top': left_top,
                "left_filename": self.left_filenames[index]}
