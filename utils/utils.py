import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from copy import deepcopy


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


def save_args(args, filename):
    with open(filename, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))


def make_dir_from_list(dirpath_list):
    for dirpath in dirpath_list:
        os.makedirs(dirpath, exist_ok=True)


def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}


def log_depth_errors(txt_path, metrics, first_line):
    print('{}'.format(first_line))
    print("abs_rel sq_rel abs_diff rmse rmse_log irmse log_10 silog a01 a02 a1 a2 a3")
    print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (
        metrics['abs_rel'], metrics['sq_rel'], metrics['abs_diff'],
        metrics['rmse'], metrics['rmse_log'], metrics['irmse'],
        metrics['log_10'], metrics['silog'],
        metrics['a01']*100.0, metrics['a02']*100.0,
        metrics['a1']*100.0, metrics['a2']*100.0, metrics['a3']*100.0))

    with open(txt_path, 'a') as f:
        f.write('{}\n'.format(first_line))
        f.write("abs_rel sq_rel abs_diff rmse rmse_log irmse log_10 silog a01 a02 a1 a2 a3\n")
        f.write("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n\n" % (
            metrics['abs_rel'], metrics['sq_rel'], metrics['abs_diff'],
            metrics['rmse'], metrics['rmse_log'], metrics['irmse'],
            metrics['log_10'], metrics['silog'],
            metrics['a01']*100.0, metrics['a02']*100.0,
            metrics['a1']*100.0, metrics['a2']*100.0, metrics['a3']*100.0))


def compute_depth_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))

    a01 = (thresh < 1.05).mean()
    a02 = (thresh < 1.10).mean()
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    abs_diff = np.mean(np.abs(gt - pred))

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    irmse = (1/gt - 1/pred) ** 2
    irmse = np.sqrt(irmse.mean())

    return dict(a01=a01, a02=a02, a1=a1, a2=a2, a3=a3,
                abs_rel=abs_rel, sq_rel=sq_rel, abs_diff=abs_diff,
                rmse=rmse, log_10=log_10, irmse=irmse,
                rmse_log=rmse_log, silog=silog)


__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
def unnormalize(img_in):
    img_out = np.zeros(img_in.shape)
    for ich in range(3):
        img_out[:, :, ich] = img_in[:, :, ich] * __imagenet_stats['std'][ich]
        img_out[:, :, ich] += __imagenet_stats['mean'][ich]
    img_out = (img_out * 255).astype(np.uint8)
    return img_out


def normal_to_rgb(norm):
    norm_rgb = ((norm + 1) * 0.5) * 255
    norm_rgb = np.clip(norm_rgb, a_min=0, a_max=255)
    norm_rgb = norm_rgb.astype(np.uint8)  # (B, H, W, 3)
    return norm_rgb


def kappa_to_alpha(pred_kappa):
    alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
            + ((np.exp(- pred_kappa * np.pi) * np.pi) / (1 + np.exp(- pred_kappa * np.pi)))
    alpha = np.degrees(alpha)
    return alpha


def make_ply_from_vertex_list(vertex_list):
    ply = ['ply', 'format ascii 1.0']
    ply += ['element vertex {}'.format(len(vertex_list))]
    ply += ['property float x', 'property float y', 'property float z',
            'property float nx', 'property float ny', 'property float nz',
            'property uchar diffuse_red', 'property uchar diffuse_green', 'property uchar diffuse_blue']
    ply += ['end_header']
    ply += vertex_list
    return '\n'.join(ply)


def save_dmap_as_ply(img, dmap, pos, target_path):
    img_2D = np.reshape(np.transpose(img, axes=[2, 0, 1]), [3, -1])
    dmap_2D = np.reshape(np.transpose(dmap, axes=[2, 0, 1]), [1, -1])

    pixel_to_ray_array_2D = np.reshape(np.transpose(pos, axes=[2, 0, 1]), [3, -1])
    pixel_to_ray_array_2D = pixel_to_ray_array_2D.astype(np.float32)

    world_coord = pixel_to_ray_array_2D * dmap_2D

    # colors
    r = deepcopy(img_2D[0, :].astype('uint8'))
    g = deepcopy(img_2D[1, :].astype('uint8'))
    b = deepcopy(img_2D[2, :].astype('uint8'))

    # coordinates
    x = deepcopy(world_coord[0, :])
    y = deepcopy(world_coord[1, :])
    z = deepcopy(world_coord[2, :])

    non_zero_idx = np.nonzero(z)
    r = r[non_zero_idx]
    g = g[non_zero_idx]
    b = b[non_zero_idx]
    x = x[non_zero_idx]
    y = y[non_zero_idx]
    z = z[non_zero_idx]

    # first ply: color-coded
    vertex_list_rgb = []
    for x_, y_, z_, r_, g_, b_ in zip(x, y, z, r, g, b):
        if z_ > 1e-3:
            vertex_list_rgb.append('{} {} {} 0 0 0 {} {} {}'.format(x_, y_, z_, r_, g_, b_))

    ply_file_rgb = open(target_path, 'w')
    ply_file_rgb.write(make_ply_from_vertex_list(vertex_list_rgb))
    ply_file_rgb.close()


# visualize during training
def visualize(args, input_dict, gt_dmap, gt_dmap_mask, pred_list, total_iter):
    d_max, e_max = 5.0, 0.5

    img = input_dict['img']
    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()[0, ...] # (H, W, 3)
    img = unnormalize(img)
    target_path = '%s/%08d_img.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, img)

    # pred_norm
    pred_norm = input_dict['pred_norm']
    pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()[0, ...] # (H, W, 3)
    target_path = '%s/%08d_pred_norm.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, normal_to_rgb(pred_norm))

    # pred_kappa
    pred_kappa = input_dict['pred_kappa']
    pred_kappa = pred_kappa.detach().cpu().permute(0, 2, 3, 1).numpy()[0, :, :, 0]
    target_path = '%s/%08d_pred_alpha.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, kappa_to_alpha(pred_kappa), vmin=0.0, vmax=60.0, cmap='jet')

    # gt depth
    gt_dmap = gt_dmap * gt_dmap_mask
    gt_dmap = gt_dmap.detach().cpu().permute(0, 2, 3, 1).numpy()             # (B, H, W, 1)
    gt_dmap = gt_dmap[0, :, :, 0]

    target_path = '%s/%08d_gt_dmap.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, gt_dmap, vmin=0.0, vmax=d_max, cmap='jet')

    # pred depth
    for i in range(len(pred_list)):
        pred_dmap = pred_list[i].detach().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 2)
        pred_dmap = pred_dmap.astype(np.float32)[0, :, :, 0]

        # pred dmap
        target_path = '%s/%08d_pred_dmap_iter%02d.jpg' % (args.exp_vis_dir, total_iter, i)
        plt.imsave(target_path, pred_dmap, vmin=0.0, vmax=d_max, cmap='jet')

        # pred emap
        pred_emap = np.abs(pred_dmap - gt_dmap)
        pred_emap[gt_dmap < args.min_depth] = 0.0
        pred_emap[gt_dmap > args.max_depth] = 0.0

        target_path = '%s/%08d_pred_emap_iter%02d.jpg' % (args.exp_vis_dir, total_iter, i)
        plt.imsave(target_path, pred_emap, vmin=0.0, vmax=e_max, cmap='Reds')
