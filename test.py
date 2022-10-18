import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utils as utils

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()

    # model architecture
    parser.add_argument('--train_data', type=str, default='scannet', help='{nyuv2, scannet}')
    parser.add_argument("--test_data", type=str, default='scannet', help="{nyuv2, scannet, custom}")

    parser.add_argument('--NNET_architecture', type=str, default=None)
    parser.add_argument('--NNET_ckpt', type=str, default=None)
    parser.add_argument('--IronDepth_ckpt', type=str, default=None)

    parser.add_argument('--train_iter', type=int, default=3)
    parser.add_argument('--test_iter', type=int, default=20)
    args = parser.parse_args()

    if args.train_data == 'scannet':
        args.NNET_architecture = 'BN'
        args.NNET_ckpt = './checkpoints/normal_scannet.pt'
        args.IronDepth_ckpt = './checkpoints/irondepth_scannet.pt'
    elif args.train_data == 'nyuv2':
        args.NNET_architecture = 'GN'
        args.NNET_ckpt = './checkpoints/normal_nyuv2.pt'
        args.IronDepth_ckpt = './checkpoints/irondepth_nyuv2.pt'

    device = torch.device('cuda:0')

    # define N_NET (surface normal estimation network)
    from models_normal.NNET import NNET
    n_net = NNET(args).to(device)
    print('loading N-Net weights from %s' % args.NNET_ckpt)
    n_net = utils.load_checkpoint(args.NNET_ckpt, n_net)
    n_net.eval()

    # define IronDepth
    from models.IronDepth import IronDepth
    model = IronDepth(args).to(device)
    print('loading IronDepth weights from %s' % args.IronDepth_ckpt)
    model = utils.load_checkpoint(args.IronDepth_ckpt, model)
    model.eval()

    # define dataloader
    from data.dataloader_custom import CustomLoader
    test_loader = CustomLoader('./examples/data/%s/' % args.test_data).data

    # output dir
    output_dir = './examples/output/%s/' % args.test_data
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for data_dict in tqdm(test_loader):
            img_name = data_dict['img_name'][0]

            img = data_dict['img'].to(device)
            pos = data_dict['pos'].to(device)

            # surface normal prediction
            norm_out = n_net(img)
            pred_norm = norm_out[:, :3, :, :]
            pred_kappa = norm_out[:, 3:, :, :]

            input_dict = {
                'img': img,
                'pred_norm': pred_norm,
                'pred_kappa': pred_kappa,
                'pos': pos,
            }

            # IronDepth forward pass
            pred_list = model(input_dict, 'test')

            # visualize

            # input image
            img = img.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)
            img = utils.unnormalize(img[0, ...])
            target_path = '%s/%s_img.png' % (output_dir, img_name)
            plt.imsave(target_path, img)

            # predicted surface normal
            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()
            target_path = '%s/%s_pred_norm.png' % (output_dir, img_name)
            plt.imsave(target_path, utils.normal_to_rgb(pred_norm[0, ...]))

            # surface normal uncertainty
            pred_kappa = pred_kappa.detach().cpu().permute(0, 2, 3, 1).numpy()
            target_path = '%s/%s_pred_norm_uncertainty.png' % (output_dir, img_name)
            plt.imsave(target_path, utils.kappa_to_alpha(pred_kappa[0, :, :, 0]), vmin=0.0, vmax=60.0, cmap='jet')

            pos = pos.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 2)
            _, H, W, _ = pos.shape
            pos_ = np.ones((H, W, 3))
            pos_[:,:,:2] = pos[0,...]

            for i in range(len(pred_list)):
                pred_dmap = pred_list[i].detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
                pred_dmap = pred_dmap[0, ...]

                # pred dmap
                target_path = '%s/%s_pred_dmap_iter%02d.png' % (output_dir, img_name, i)
                plt.imsave(target_path, pred_dmap[:, :, 0], cmap='jet')

                if i == args.test_iter:
                    target_path = '%s/%s_pred_dmap_iter%02d.ply' % (output_dir, img_name, i)    
                    utils.save_dmap_as_ply(img, pred_dmap, pos_, target_path)

