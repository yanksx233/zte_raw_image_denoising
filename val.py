import os
import random
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.log import get_logger
from utils.dataset import TrainSet
from model.build import build_model
from utils.util import cal_psnr_and_ssim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, KFold


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser('test processing')

    # TODO:
    parser.add_argument('--model', type=str, default='swin')
    parser.add_argument('--drop_path_rate', type=float, default=0)
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('--select_fold', type=int, default=5)
    parser.add_argument('--n_splits', type=int, default=300)
    parser.add_argument('--use_checkpoint', action='store_true')
    # --------------------

    # dir setting
    parser.add_argument('--dataset_dir', type=str, default='../Dataset/withsidd_uint16_400x400_fold_0/val/')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--save_dir', type=str, default='demo')  # ./log/save_dir/test/

    # device
    parser.add_argument('--device', type=str, default='cuda:0')

    # test setting
    parser.add_argument('--start_iter', type=int, default=50000)
    parser.add_argument('--end_iter', type=int, default=100000)
    parser.add_argument('--step', type=int, default=5000)

    # mode
    parser.add_argument('--mode', type=str, default='val')

    args = parser.parse_args()
    return args


def update_state_dict(model, pretrained_dict):
    own_state_dict = model.state_dict()
    # filter parameters
    pretrained_dict = {name: param for name, param in pretrained_dict.items() if name in own_state_dict and 'mask' not in name}
    own_state_dict.update(pretrained_dict)
    model.load_state_dict(own_state_dict)


def test(model, logger, test_loader, device):
    model.eval()
    with torch.no_grad():
        psnr_list, ssim_list = [], []
        for idx, sample in enumerate(test_loader):
            lq, gt = sample['lq'].to(device), sample['gt'].to(device)
            im_rec = model(lq)

            im_rec = torch.clamp(im_rec, 0, 1)
            psnr, ssim = cal_psnr_and_ssim(im_rec, gt)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

        psnr = float(np.array(psnr_list).mean())
        ssim = float(np.array(ssim_list).mean())

        logger.debug(f'The average psnr: {psnr:.5f}')
        logger.debug(f'The average ssim: {ssim:.5f}')

        return psnr, ssim


if __name__ == '__main__':
    args = get_args()
    set_random_seed(233)
    logger = get_logger(args)
    
    iters = np.arange(args.start_iter, args.end_iter+1, args.step)

    model = build_model(args).to(args.device)
    torch.backends.cudnn.benchmark = True
    
    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    logger.debug("Number of parameter: %.2fM" % (total / 1e6))
    logger.info(f'Model type: {args.model}')

    # dataset = TrainSet(args.dataset_dir, train=False)
    # idx = np.array(range(len(dataset)))
    # kf = KFold(n_splits=args.n_splits, shuffle=True)
    # for fold, (train_idx, val_idx) in enumerate(kf.split(idx)):
    #     if fold == args.select_fold - 1:
    #         val_set = Subset(dataset, val_idx)
    #         break
    val_set = TrainSet(args.dataset_dir, train=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    psnr_iter = []
    ssim_iter = []
    for idx, iter in enumerate(iters):
        model_file = os.path.join(args.log_dir, args.save_dir, 'train/checkpoint', f'iter_{int(iter/1000)}k.tar')
        logger.info(f'Load model: {model_file}...')
        checkpoint = torch.load(model_file, map_location=args.device)
        update_state_dict(model, checkpoint['model_state_dict'])

        current_psnr, current_ssim = test(model, logger, val_loader, args.device)
        psnr_iter.append(current_psnr)
        ssim_iter.append(current_ssim)
        logger.debug('-----------------------------------------------------------\n')

    logger.debug('===========================================================')

    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(iters, psnr_iter, label='PSNR', c='r')
    ax1.scatter(iters, psnr_iter, c='c')
    ax1.set_xticks(iters)
    # ax1.tick_params(labelsize=22)
    ax1.set_xlabel('iter', fontsize=22)
    ax1.set_ylabel('PSNR', fontsize=22)
    ax1.set_title(f'select_fold={args.select_fold}', fontsize=24)

    ax2 = ax1.twinx()
    ax2.plot(iters, ssim_iter, label='SSIM', c='b')
    ax2.scatter(iters, ssim_iter, c='m')
    ax2.set_ylabel('SSIM', fontsize=22)
    # ax2.tick_params(labelsize=22)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=4, fontsize=14)

    fig.savefig(os.path.join(args.log_dir, args.save_dir, args.mode, 'result.jpg'))
    
    logger.info(f'Best PSNR: {np.max(psnr_iter)}, iter: {iters[np.argmax(psnr_iter)]}')
    logger.info(f'Best SSIM: {np.max(ssim_iter)}, iter: {iters[np.argmax(ssim_iter)]}')
    logger.debug('===========================================================\n')

    
