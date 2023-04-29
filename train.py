import argparse
import os
import sys
import random
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed

import utils.utils as utils
from utils.losses import compute_loss


def train(model, args, device):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    should_write = ((not args.distributed) or args.rank == 0)

    # dataloader
    if args.dataset_name == 'scannet':
        from data.dataloader_scannet import ScannetLoader
        train_loader = ScannetLoader(args, 'train').data
        test_loader = ScannetLoader(args, 'test').data
    else:
        raise Exception

    # define losses
    loss_fn = compute_loss(args)

    # optimizer
    m = model.module if args.multigpu else model
    params = [{"params": m.get_1x_lr_params(), "lr": args.lr / 10},
                {"params": m.get_10x_lr_params(), "lr": args.lr}]
    optimizer = optim.AdamW(params, weight_decay=args.weight_decay, lr=args.lr)

    # learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                              max_lr=args.lr,
                                              epochs=args.n_epochs,
                                              steps_per_epoch=len(train_loader))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()
       
    # start training
    total_iter = 0
    model.train()
    for epoch in range(args.n_epochs):
        if args.rank == 0:
            t_loader = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{args.n_epochs}. Loop: Train",
                            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', total=len(train_loader))
        else:
            t_loader = train_loader

        for data_dict in t_loader:
            optimizer.zero_grad()
            total_iter += args.batch_size_orig

            # data to device
            img = data_dict['img'].to(device)                   # (B, 3, H, W)
            gt_dmap = data_dict['depth_gt'].to(device)          # (B, 1, H, W)
            pred_norm = data_dict['pred_norm'].to(device)       # (B, 3, H, W)
            pred_kappa = data_dict['pred_kappa'].to(device)     # (B, 1, H, W)
            pos = data_dict['pos'].to(device)                   # (B, 2, H, W)

            input_dict = {
                'img': img,
                'pred_norm': pred_norm,
                'pred_kappa': pred_kappa,
                'pos': pos,
            }

            # gt dmap mask
            gt_dmap[gt_dmap > args.max_depth] = 0.0
            gt_dmap_mask = gt_dmap > args.min_depth

            # forward pass
            pred_list = model(input_dict, 'train')

            # compute loss
            loss = loss_fn(pred_list, gt_dmap, gt_dmap_mask)

            # display loss
            loss_ = float(loss.data.cpu().numpy())
            if args.rank == 0:
                t_loader.set_description(f"Epoch: {epoch + 1}/{args.n_epochs}. Loop: Train. Loss: {'%.5f' % loss_}")
                t_loader.refresh()

            # back-propagate
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # lr scheduler
            scheduler.step()

        # visualization and validation
        if should_write:
            utils.visualize(args, input_dict, gt_dmap, gt_dmap_mask, pred_list, total_iter)
            
            model.eval()
            metrics = validate(model, args, test_loader, device)
            utils.log_depth_errors(args.eval_acc_txt, metrics, 'total_iter: {}'.format(total_iter))
            target_path = args.exp_model_dir + '/checkpoint_iter_%010d.pt' % total_iter
            print(target_path)
            torch.save({"model": model.state_dict(),
                        "iter": total_iter}, target_path)
            model.train()

    return model


def validate(model, args, test_loader, device='cpu'):
    with torch.no_grad():
        metrics = utils.RunningAverageDict()

        for data_dict in tqdm(test_loader, desc=f"Loop: Validation", 
                                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', total=len(test_loader)):

            img = data_dict['img'].to(device)                   # (B, 3, H, W)
            gt_dmap = data_dict['depth_gt'].to(device)          # (B, 1, H, W)
            pred_norm = data_dict['pred_norm'].to(device)       # (B, 3, H, W)
            pred_kappa = data_dict['pred_kappa'].to(device)     # (B, 1, H, W)
            pos = data_dict['pos'].to(device)                   # (B, 2, H, W)

            input_dict = {
                'img': img,
                'pred_norm': pred_norm,
                'pred_kappa': pred_kappa,
                'pos': pos,
                'init_dmap': None
            }

            # forward pass
            pred_list = model(input_dict, 'test')

            gt_dmap = gt_dmap.detach().cpu().permute(0, 2, 3, 1).numpy()    # (B, H, W, 1)
            pred_dmap = pred_list[-1].detach().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 2)
            gt_dmap = gt_dmap[0, :, :, 0]
            pred_dmap = pred_dmap[0, :, :, 0]
            valid_mask = np.logical_and(gt_dmap > args.min_depth, gt_dmap < args.max_depth)

            # masking
            pred_dmap[pred_dmap < args.min_depth] = args.min_depth
            pred_dmap[pred_dmap > args.max_depth] = args.max_depth
            pred_dmap[np.isinf(pred_dmap)] = args.max_depth
            pred_dmap[np.isnan(pred_dmap)] = args.min_depth

            metrics.update(utils.compute_depth_errors(gt_dmap[valid_mask], pred_dmap[valid_mask]))

        return metrics.get_value()
        

# main worker
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    # define model
    from models.IronDepth import IronDepth
    model = IronDepth(args)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        # print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    train(model, args, device=args.gpu)


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    # directory
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--train_iter', type=int, default=3)         # iteration (train)
    parser.add_argument('--test_iter', type=int, default=10)         # iteration (test)
    parser.add_argument('--loss_gamma', type=float, default=0.8)

    # training
    parser.add_argument('--n_epochs', default=5, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument("--distributed", default=True, action="store_true", help="Use DDP if set")
    parser.add_argument("--workers", default=4, type=int, help="Number of workers for data loading")

    # optimizer setup
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
    parser.add_argument('--lr', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--grad_clip', default=1.0, type=float)

    # dataset
    parser.add_argument("--dataset_name", type=str, default='scannet')
    parser.add_argument('--input_height', type=int, default=480)
    parser.add_argument('--input_width', type=int, default=640)
    parser.add_argument('--crop_height', type=int, default=416)
    parser.add_argument('--crop_width', type=int, default=544)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)

    # dataset - augmentation
    parser.add_argument("--data_augmentation_color", default=True, action="store_true")
    parser.add_argument("--data_augmentation_flip", default=True, action="store_true")
    parser.add_argument("--data_augmentation_crop", default=True, action="store_true")

    # read arguments from txt file
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.num_threads = args.workers
    args.mode = 'train'

    # create experiment directory
    args.exp_dir = './exp/%s' % args.exp_name
    args.exp_model_dir = args.exp_dir + '/models/'    # store model checkpoints
    args.exp_vis_dir = args.exp_dir + '/vis/'         # store training images
    args.exp_log_dir = args.exp_dir + '/log/'         # store log
    utils.make_dir_from_list([args.exp_dir, args.exp_model_dir, args.exp_vis_dir, args.exp_log_dir])

    # set up logging
    utils.save_args(args, args.exp_log_dir + '/params.txt')  # save experiment parameters
    args.eval_acc_txt = args.exp_log_dir + '/eval_acc.txt'          # metric accuracy

    # train
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    args.world_size = 1
    args.rank = 0
    nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    args.batch_size_orig = args.batch_size

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
