import argparse
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import random

from dataloader.ds_loader import DSDataset
from dataloader.kitti_loader import KITTIDataset
from dataloader.driving_loader import DrivingDataset
from networks.stackhourglass import PSMNet


class RandomDataLoaderSampler:
    def __init__(self, dataloaders, shuffle=False):
        self.dataloaders = dataloaders
        self.active_loader = None
        self.shuffle = shuffle

    def _reset_iter(self):
        self.active_loader = [iter(loader) for loader in self.dataloaders]

    def __len__(self):
        return sum([len(x) for x in self.dataloaders])

    def __iter__(self):
        self._reset_iter()
        return self

    def __next__(self):
        next_batch = None
        while next_batch is None:
            if len(self.active_loader) == 0:
                raise StopIteration

            loader_idx = 0 if not self.shuffle else random.randint(0, len(self.active_loader) - 1)
            try:
                next_batch = next(self.active_loader[loader_idx])
            except StopIteration:
                del self.active_loader[loader_idx]

        return next_batch


parser = argparse.ArgumentParser(description='LaC')
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='0,1,2,3')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--load_path', type=str,
                    default='/work/vig/tianyed/Lac-GwcNet/Lac-GwcNet/state_dicts/SceneFlow.pth')
parser.add_argument('--save_path', type=str, default='/scratch/ding.tian/logs_ddp/Lac-Gwc_ft/')
parser.add_argument('--max_disp', type=int, default=192)
parser.add_argument('--lsp_width', type=int, default=3)
parser.add_argument('--lsp_height', type=int, default=3)
parser.add_argument('--lsp_dilation', type=list, default=[1, 2, 4, 8])
parser.add_argument('--lsp_mode', type=str, default='separate')
parser.add_argument('--lsp_channel', type=int, default=4)
parser.add_argument('--no_udc', action='store_true', default=False)
parser.add_argument('--refine', type=str, default='csr')
args = parser.parse_args()

if not args.no_cuda:
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
cuda = torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

ds_train_dataset = DSDataset('/work/vig/Datasets/DrivingStereo/',
                             '/work/vig/tianyed/Lac-GwcNet/Lac-GwcNet/filenames/DS_train.txt', True)
driving_train_dataset = DrivingDataset('/work/vig/Datasets/SyntheticDriving/',
                                       '/work/vig/tianyed/Lac-GwcNet/Lac-GwcNet/filenames/Driving_vox_train.txt', True)
kitti_train_dataset = KITTIDataset('/work/vig/Datasets/KITTI_VoxelFlow/',
                                   '/work/vig/tianyed/Lac-GwcNet/Lac-GwcNet/filenames/KITTI_vox_train.txt', True)

ds_test_dataset = DSDataset('/work/vig/Datasets/DrivingStereo/',
                            '/work/vig/tianyed/Lac-GwcNet/Lac-GwcNet/filenames/DS_test.txt', False)
driving_test_dataset = DrivingDataset('/work/vig/Datasets/SyntheticDriving/',
                                      '/work/vig/tianyed/Lac-GwcNet/Lac-GwcNet/filenames/Driving_vox_test.txt', False)
kitti_test_dataset = KITTIDataset('/work/vig/Datasets/KITTI_VoxelFlow/',
                                  '/work/vig/tianyed/Lac-GwcNet/Lac-GwcNet/filenames/KITTI_vox_test.txt', False)

ds_trainLoader = torch.utils.data.DataLoader(ds_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16,
                                             drop_last=True)
driving_trainLoader = torch.utils.data.DataLoader(driving_train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=16, drop_last=True)
kitti_trainLoader = torch.utils.data.DataLoader(kitti_train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=16, drop_last=True)
TrainImgLoader = RandomDataLoaderSampler([ds_trainLoader, driving_trainLoader, kitti_trainLoader], shuffle=True)

ds_testLoader = torch.utils.data.DataLoader(ds_test_dataset, batch_size=4, shuffle=False, num_workers=16,
                                            drop_last=False)
driving_testLoader = torch.utils.data.DataLoader(driving_test_dataset, batch_size=4, shuffle=False, num_workers=16,
                                                 drop_last=False)
kitti_testLoader = torch.utils.data.DataLoader(kitti_test_dataset, batch_size=4, shuffle=False, num_workers=16,
                                               drop_last=False)
TestImgLoader = RandomDataLoaderSampler([ds_testLoader, driving_testLoader, kitti_testLoader], shuffle=False)

affinity_settings = {}
affinity_settings['win_w'] = args.lsp_width
affinity_settings['win_h'] = args.lsp_width
affinity_settings['dilation'] = args.lsp_dilation
udc = not args.no_udc

model = PSMNet(maxdisp=args.max_disp, struct_fea_c=args.lsp_channel, fuse_mode=args.lsp_mode,
               affinity_settings=affinity_settings, udc=udc, refine=args.refine)
model = nn.DataParallel(model)
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
if cuda:
    model.cuda()

checkpoint = torch.load(args.load_path)
model.load_state_dict(checkpoint)

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))


def train(imgL, imgR, disp_true):
    model.train()
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    disp_true = torch.FloatTensor(disp_true)

    if cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    optimizer.zero_grad()

    loss1, loss2 = model(imgL, imgR, disp_true)
    loss1 = torch.mean(loss1)
    loss2 = torch.mean(loss2)

    if udc:
        loss = 0.1 * loss1 + loss2
    else:
        loss = loss1

    loss.backward()
    optimizer.step()

    return loss.item()


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)

    if cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        pred_disp = model(imgL, imgR, torch.zeros_like(disp_true).cuda())

    final_disp = pred_disp.cpu()
    true_disp = disp_true
    index = np.argwhere(true_disp > 0)
    disp_true[index[0], index[1], index[2]] = np.abs(
        true_disp[index[0], index[1], index[2]] - final_disp[index[0], index[1], index[2]])
    correct = (disp_true[index[0], index[1], index[2]] < 3) | \
              (disp_true[index[0], index[1], index[2]] < true_disp[index[0], index[1], index[2]] * 0.05)

    torch.cuda.empty_cache()

    return 1 - (float(torch.sum(correct)) / float(len(index[0])))


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = 0.001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_epoch = 1

    for epoch in range(start_epoch, args.epoch + start_epoch):
        print('This is %d-th epoch' % epoch)
        total_train_loss = 0
        total_test_loss = 0
        adjust_learning_rate(optimizer, epoch)

        for batch_id, sample in enumerate(tqdm(TrainImgLoader)):
            imgL, imgR, disp_L = sample['left'], sample['right'], sample['disparity']
            train_loss = train(imgL, imgR, disp_L)
            total_train_loss += train_loss
        avg_train_loss = total_train_loss / len(TrainImgLoader)
        print('Epoch %d average training loss = %.3f' % (epoch, avg_train_loss))

        for batch_id, sample in enumerate(tqdm(TestImgLoader)):
            imgL, imgR, disp_L = sample['left'], sample['right'], sample['disparity']
            test_loss = test(imgL, imgR, disp_L)
            total_test_loss += test_loss
        avg_test_loss = total_test_loss / len(TestImgLoader)
        print('Epoch %d total test loss = %.3f' % (epoch, avg_test_loss))

        state = {'net': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}

        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        save_model_path = args.save_path + 'checkpoint_{}.ckpt'.format(epoch)
        torch.save(state, save_model_path)

        torch.cuda.empty_cache()

    print('Training Finished!')


if __name__ == '__main__':
    main()
