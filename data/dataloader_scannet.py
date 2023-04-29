# dataloader for ScanNet
import numpy as np
from PIL import Image
import random

import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

import os
import glob


def color_augmentation(image):
    # gamma
    gamma = random.uniform(0.9, 1.1)
    image_aug = image ** gamma

    # brightness
    brightness = random.uniform(0.75, 1.25)
    image_aug = image_aug * brightness

    # color
    colors = np.random.uniform(0.9, 1.1, size=3)
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    image_aug = np.clip(image_aug, 0, 1)
    return image_aug


class ScannetLoader(object):
    def __init__(self, args, mode):
        self.t_samples = ScannetLoadPreprocess(args, mode)

        if mode == 'train':
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.t_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.t_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   drop_last=True,
                                   sampler=self.train_sampler)

        else:
            self.data = DataLoader(self.t_samples, 1, shuffle=False, num_workers=1)                                


class ScannetLoadPreprocess(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.filenames = glob.glob('./scannet/%s/scene*/*_img.png' % mode)
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # img resolution
        self.img_H = args.input_height  # 480
        self.img_W = args.input_width   # 640
        self.crop_H = args.crop_height  # 416
        self.crop_W = args.crop_width   # 544

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        depth_path = img_path.replace('_img.png', '_depth.png')
        pred_norm_path = img_path.replace('_img.png', '_norm.png')
        pred_kappa_path = img_path.replace('_img.png', '_kappa.png')
        intrins_path = os.path.join(os.path.split(img_path)[0], 'intrins.txt')
        assert os.path.exists(depth_path)
        assert os.path.exists(intrins_path)
        assert os.path.exists(pred_norm_path)
        assert os.path.exists(pred_kappa_path)

        # read img and depth
        img = Image.open(img_path).convert("RGB").resize(size=(self.img_W, self.img_H), resample=Image.BILINEAR)
        depth_gt = Image.open(depth_path).resize(size=(self.img_W, self.img_H), resample=Image.NEAREST)
        pred_norm = Image.open(pred_norm_path).convert("RGB").resize(size=(self.img_W, self.img_H), resample=Image.BILINEAR)
        pred_kappa = Image.open(pred_kappa_path).resize(size=(self.img_W, self.img_H), resample=Image.BILINEAR)

        fx, fy, cx, cy = [float(i) for i in open(intrins_path).readlines()[0].split(' ')]
        intrins = np.eye(3).astype(np.float32)
        intrins[0, 0] = fx
        intrins[1, 1] = fy
        intrins[0, 2] = cx
        intrins[1, 2] = cy

        pos = self.get_pos(intrins)

        if self.mode == 'train':
            # data augmentation - flip
            DA_flip = False
            if self.args.data_augmentation_flip:
                DA_flip = random.random() > 0.5
                if DA_flip:
                    img = TF.hflip(img)
                    depth_gt = TF.hflip(depth_gt)
                    pred_norm = TF.hflip(pred_norm)
                    pred_kappa = TF.hflip(pred_kappa)
                    pos = TF.hflip(pos)

            # img to array
            img = np.array(img).astype(np.float32) / 255.0

            # depth to array
            depth_gt = np.array(depth_gt)[:, :, np.newaxis].astype(np.float32)  # (H, W, 1)
            depth_gt = depth_gt / 1000.0    # convert to meters

            # norm to array
            pred_norm = np.array(pred_norm).astype(np.uint8)
            pred_norm = ((np.array(pred_norm).astype(np.float32) / 255.0) * 2.0) - 1.0

            # make sure to flip the signs
            if DA_flip:
                pred_norm[:, :, 0] = - pred_norm[:, :, 0]
                pos[0, :, :] = - pos[0, :, :]

            # kappa to array
            pred_kappa = np.array(pred_kappa)[:, :, np.newaxis].astype(np.float32)
            pred_kappa = pred_kappa / 256.0

            # data augmentation - random crop
            if self.args.data_augmentation_crop:
                img, depth_gt, pred_norm, pred_kappa, pos = \
                    self.random_crop(img, depth_gt, pred_norm, pred_kappa, pos)

            # data augmentation - color
            if self.args.data_augmentation_color:
                if random.random() > 0.5:
                    img = color_augmentation(img)

        else:
            img = np.array(img).astype(np.float32) / 255.0

            depth_gt = np.array(depth_gt)[:, :, np.newaxis].astype(np.float32)  # (H, W, 1)
            depth_gt = depth_gt / 1000.0

            pred_norm = np.array(pred_norm).astype(np.uint8)
            pred_norm = ((np.array(pred_norm).astype(np.float32) / 255.0) * 2.0) - 1.0

            pred_kappa = np.array(pred_kappa)[:, :, np.newaxis].astype(np.float32)
            pred_kappa = pred_kappa / 256.0

        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        img = self.normalize(img)
        depth_gt = torch.from_numpy(depth_gt).permute(2, 0, 1)  # (1, H, W)
        pred_norm = torch.from_numpy(pred_norm).permute(2, 0, 1)  # (3, H, W)
        pred_kappa = torch.from_numpy(pred_kappa).permute(2, 0, 1)  # (1, H, W)

        sample = {
            'img': img,
            'depth_gt': depth_gt,
            'pred_norm': pred_norm,
            'pred_kappa': pred_kappa,
            'pos': pos, 
            'intrins': torch.from_numpy(intrins),
            'img_path': img_path,
        }

        return sample

    def get_pos(self, intrins):
        W, H = 640, 480
        pos = np.ones((H, W, 2))
        x_range = np.concatenate([np.arange(W).reshape(1, W)] * H, axis=0)
        y_range = np.concatenate([np.arange(H).reshape(H, 1)] * W, axis=1)
        pos[:, :, 0] = x_range + 0.5
        pos[:, :, 1] = y_range + 0.5
        pos[:, :, 0] = np.arctan((pos[:, :, 0] - intrins[0, 2]) / intrins[0, 0])
        pos[:, :, 1] = np.arctan((pos[:, :, 1] - intrins[1, 2]) / intrins[1, 1])
        pos = torch.from_numpy(pos.astype(np.float32)).permute(2, 0, 1)
        return pos        

    def random_crop(self, img, depth, pred_norm, pred_kappa, pos):
        height, width = self.crop_H, self.crop_W
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        pred_norm = pred_norm[y:y + height, x:x + width, :]
        pred_kappa = pred_kappa[y:y + height, x:x + width, :]
        pos = pos[:, y:y + height, x:x + width]
        return img, depth, pred_norm, pred_kappa, pos

