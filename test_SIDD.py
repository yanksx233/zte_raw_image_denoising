import os
import time
import random
import torch
import argparse
import numpy as np
import pandas as pd
from utils.log import get_logger
from utils.dataset import SIDD, ToTensor
from model.build import build_model
from utils.util import cal_psnr_and_ssim


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser('test processing')

    # TODO:
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--model', type=str, default='restormer')
    parser.add_argument('--drop_path_rate', type=float, default=0)
    parser.add_argument('--use_mask', action='store_true')
    # --------------------

    # dir setting
    parser.add_argument('--dataset_dir', type=str, default='../Dataset/SIDD_Medium_Raw/Data')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--save_dir', type=str, default='test_SIDD')  # ./log/save_dir/test/
    parser.add_argument('--pretrain', type=str, default='./log/restormer_150k_cutmixup_comp/train/checkpoint/iter_95k.tar')

    # device
    parser.add_argument('--device', type=str, default='cpu')

    # mode
    parser.add_argument('--mode', type=str, default='test')

    args = parser.parse_args()
    return args


def update_state_dict(model, pretrained_dict):
    own_state_dict = model.state_dict()
    # filter parameters
    pretrained_dict = {name: param for name, param in pretrained_dict.items() if name in own_state_dict and 'mask' not in name}
    own_state_dict.update(pretrained_dict)
    model.load_state_dict(own_state_dict)


@torch.no_grad()
def test(model, logger, dataset, device):
    model.eval()
    data = {'dir': [], 'idx': [], 'psnr': [], 'ssim': [], 'time': []}
    for idx, (sample, dir_name) in enumerate(dataset):
        sample = ToTensor()(sample, norm=False)
        lq, gt = sample['lq'].to(device)[None], sample['gt'].to(device)[None]

        t = time.time()
        im_rec = model(lq)
        t = time.time() - t

        im_rec = torch.clamp(im_rec, 0, 1)
        psnr, ssim = cal_psnr_and_ssim(im_rec, gt)

        logger.debug(f'{idx+1}/{len(dataset)},  dir: {dir_name},  idx: {idx % 2},  psnr: {psnr:.5f},  ssim: {ssim:.5f},  time: {t/60.:.2f} min\n')

        data['dir'].append(dir_name)
        data['idx'].append(idx % 2)
        data['psnr'].append(psnr)
        data['ssim'].append(ssim)
        data['time'].append(f'{t/60.:.2f} min')

    df = pd.DataFrame(data)
    score = (0.8 * (df['psnr'] - 30) / 30 + 0.2 * (df['ssim'] - 0.8) / 0.2) * 100
    df['score'] = score
    df.to_csv(os.path.join(args.log_dir, args.save_dir, 'test', 'sidd.csv'))

    return psnr, ssim


if __name__ == '__main__':
    args = get_args()
    set_random_seed(233)
    logger = get_logger(args)
    
    model = build_model(args).to(args.device)
    torch.backends.cudnn.benchmark = True
    
    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    logger.debug("Number of parameter: %.2fM" % (total / 1e6))
    logger.debug(f'Model type: {args.model}')

    dataset = SIDD(args.dataset_dir)

    logger.info(f'Load model: {args.pretrain}...')
    checkpoint = torch.load(args.pretrain, map_location=args.device)
    update_state_dict(model, checkpoint['model_state_dict'])

    test(model, logger, dataset, args.device)

    # Generate the .csv file from the .log file.
    # with open(os.path.join(args.log_dir, args.save_dir, 'test', 'test.log'), 'r') as f:
    #     lines = f.readlines()
    
    # lines = lines[3:]
    # data = {'dir': [], 'idx': [], 'psnr': [], 'ssim': [], 'time': []}
    # for line in lines:
    #     line = line.rstrip().replace(' ', '').split(',')[2:]
    #     for kv in line:
    #         k, v = kv.split(':')
    #         data[k].append(v)

    # df = pd.DataFrame(data)
    # df[['psnr', 'ssim']] = df[['psnr', 'ssim']].astype('float')
    # score = (0.8 * (df['psnr'] - 30) / 30 + 0.2 * (df['ssim'] - 0.8) / 0.2) * 100
    # df['score'] = score
    
    # df.to_csv(os.path.join(args.log_dir, args.save_dir, 'test', 'sidd.csv'))

