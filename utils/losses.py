import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class compute_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args.loss_gamma

    def forward(self, pred_list, gt_depth, gt_depth_mask):
        n_predictions = len(pred_list)
        loss = 0.0

        for i in range(n_predictions):
            pred = pred_list[i]
            i_weight = self.gamma ** (n_predictions - i - 1)
            loss = loss + i_weight * self.l1_loss(pred, gt_depth, gt_depth_mask)

        return loss

    def l1_loss(self, out, gt_depth, gt_depth_mask):
        gt_depth = gt_depth[gt_depth_mask]
        pred_depth = out[gt_depth_mask]
        l1 = torch.abs(pred_depth - gt_depth)
        return torch.mean(l1)
        