# dataloader
import numpy as np
from PIL import Image

import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

import os
import glob


class CustomLoader(object):
    def __init__(self, dataset_path):
        self.t_samples = CustomLoadPreprocess(dataset_path)
        self.data = DataLoader(self.t_samples, 1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=False)


class CustomLoadPreprocess(Dataset):
    def __init__(self, dataset_path):
        self.filenames = glob.glob(dataset_path + '/*.png') + glob.glob(dataset_path + '/*.jpg')
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        ext = os.path.splitext(img_path)[-1]
        img_name = img_path.split('/')[-1].split(ext)[0]

        # read image
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # read camera intrinsics (if exists)
        intrins_path = img_path.replace(ext, '.txt')

        if os.path.exists(intrins_path):
            with open(intrins_path, 'r') as f:
                content = f.readlines()
            content = [float(x.strip()) for x in content]
            fx, fy, cx, cy = content

        elif 'nyuv2' in intrins_path:
            # camera intrinsics (constant for NYUv2)
            fx = 518.85790117450188
            fy = 519.46961112127485
            cx = 325.58244941119034 - 0.5
            cy = 253.73616633400465 - 0.5

        else:
            width = int(round(width / 16) * 16)
            height = int(round(height / 16) * 16)
            img = img.resize(size=(width, height), resample=Image.BILINEAR)

            fx = fy = 500.0
            cx = width // 2
            cy = height // 2

        intrins = np.eye(3).astype(np.float32)
        intrins[0, 0] = fx
        intrins[1, 1] = fy
        intrins[0, 2] = cx
        intrins[1, 2] = cy

        # pos is an array of rays with unit depth
        pos = self.get_pos(intrins, width, height)

        img = np.array(img).astype(np.float32) / 255.0
        img = self.normalize(torch.from_numpy(img).permute(2, 0, 1))            # (3, H, W)

        sample = {'img': img,
                  'pos': pos, 
                  'intrins': torch.from_numpy(intrins),
                  'img_name': img_name}

        return sample

    def get_pos(self, intrins, W, H):
        pos = np.ones((H, W, 2))
        x_range = np.concatenate([np.arange(W).reshape(1, W)] * H, axis=0)
        y_range = np.concatenate([np.arange(H).reshape(H, 1)] * W, axis=1)
        pos[:, :, 0] = x_range + 0.5
        pos[:, :, 1] = y_range + 0.5
        pos[:, :, 0] = np.arctan((pos[:, :, 0] - intrins[0, 2]) / intrins[0, 0])
        pos[:, :, 1] = np.arctan((pos[:, :, 1] - intrins[1, 2]) / intrins[1, 1])
        pos = torch.from_numpy(pos.astype(np.float32)).permute(2, 0, 1)
        return pos        


