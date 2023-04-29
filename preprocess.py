import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import utils.utils as utils

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


def norm_to_rgb(pred_norm):
    pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
    pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
    pred_norm_rgb = pred_norm_rgb.astype(np.uint8)  # (B, H, W, 3)
    return pred_norm_rgb


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()

    # model architecture
    parser.add_argument('--NNET_architecture', type=str, default='BN')
    parser.add_argument('--NNET_ckpt', type=str, default='./checkpoints/normal_scannet.pt')
    args = parser.parse_args()
    device = torch.device('cuda:0')

    # define N_NET (surface normal estimation network)
    from models_normal.NNET import NNET
    n_net = NNET(args).to(device)
    print('loading N-Net weights from %s' % args.NNET_ckpt)
    n_net = utils.load_checkpoint(args.NNET_ckpt, n_net)
    n_net.eval()

    # generate normal predictions
    # note that the images should be of size W=640, H=480
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    with torch.no_grad():

        for mode in ['train', 'test']:
            img_paths = glob.glob('./scannet/%s/*/*img.png' % mode)
            for img_path in tqdm(img_paths):
                img = Image.open(img_path).convert("RGB")
                img = np.array(img).astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1)    # (3, H, W)
                img = normalize(img).unsqueeze(0).to(device)    # (1, 3, H, W)

                # surface normal prediction
                norm_out = n_net(img)
                pred_norm = norm_out[:, :3, :, :]
                pred_kappa = norm_out[:, 3:, :, :]
                pred_norm = pred_norm.cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)
                pred_kappa = pred_kappa.cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 1)

                # save pred normal
                target_path = img_path.replace('_img.png', '_norm.png')
                if not os.path.exists(target_path):
                    plt.imsave(target_path, norm_to_rgb(pred_norm)[0, :, :, :])
                    print(target_path)

                # save pred kappa
                target_path = img_path.replace('_img.png', '_kappa.png')
                if not os.path.exists(target_path):
                    pred_kappa = (pred_kappa * 256.).astype(np.uint16)
                    cv2.imwrite(target_path, pred_kappa[0, :, :, 0])
                    print(target_path)
