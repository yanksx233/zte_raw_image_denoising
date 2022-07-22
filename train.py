import os
import time
import torch
import random
import argparse
import numpy as np
from utils.log import get_logger
from utils.dataset import MoA, TrainSet
from model.build import build_model
from utils.util import build_optimizer, build_scheduler
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter   
from sklearn.model_selection import KFold


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser('train processing')
    
    # model
    parser.add_argument('--arch', type=str, default='restormer')
    parser.add_argument('--use_mask', action='store_true')  # swin
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--drop_path_rate', type=float, default=0.1) # swin

    # datasets
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--dataset_dir', type=str, default='../Dataset/withsidd_score50_uint16_528x528_all/train/')

    parser.add_argument('--noise_prob', type=float, default=0.25)
    parser.add_argument('--gP_prob', type=float, default=1, help='g_prob = 1 - gP_prob')

    parser.add_argument('--prob', type=float, default=0.5, help='Probability for MoA')
    parser.add_argument('--mixup', type=float, default=0)
    parser.add_argument('--cutmix', type=float, default=0)
    parser.add_argument('--cutmixup', type=float, default=1)
    parser.add_argument('--cutblur', type=float, default=0)

    # log
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--log_every_iter', type=int, default=400)
    parser.add_argument('--save_dir', type=str, default='demo')   # log_dir/{args.mode}/save_dir
    parser.add_argument('--save_every_iter', type=int, default=5000)
    parser.add_argument('--tb_log', action='store_true')

    # device
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=4)

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='for AdamW')
    parser.add_argument('--beta2', type=float, default=0.999, help='for AdamW')

    # scheduler
    parser.add_argument('--scheduler_name', type=str, default='cosine')   # choices = ['cosine', 'step']
    parser.add_argument('--periods', type=str, default='5e4,1e5', help='for cosine learning rate')
    parser.add_argument('--min_lrs', type=str, default='1e-4,1e-6', help='for cosine learning rate')
    parser.add_argument('--decay_step', type=int, default=50000, help='for step learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='for step learning rate')
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--warmup_lr', type=float, default=1e-6)

    # train
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--current_epoch', type=int, default=1)
    parser.add_argument('--current_iter', type=int, default=0)
    parser.add_argument('--total_iter', type=int, default=150000)
    parser.add_argument('--resume_from', type=str, help='continue to train from checkpoint', default=None)
    parser.add_argument('--load_from', type=str, help='load only model parameters from checkpoint', default=None)

    # mode
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()

    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)

    args.periods = list(map(lambda x: int(float(x)), args.periods.split(',')))
    args.min_lrs = list(map(float, args.min_lrs.split(',')))

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.gpu_ids[args.local_rank]) if len(args.gpu_ids) != 0 else torch.device('cpu')
        torch.backends.cudnn.benchmark = True
        
        if len(args.gpu_ids) > 1:
            torch.cuda.set_device(self.device)
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = -1
            self.world_size = 1

        # dataset = TrainSet(args.dataset_dir, args.crop_size, train=True)
        # idx = np.array(range(len(dataset)))
        # kf = KFold(n_splits=args.n_splits, shuffle=True)
        # for fold, (train_idx, val_idx) in enumerate(kf.split(idx)):
        #     if fold == args.select_fold - 1:
        #         trainset = Subset(dataset, train_idx)
        #         # valset = Subset(dataset, val_idx)
        #         break
        trainset = TrainSet(args.dataset_dir, args.crop_size, train=True, noise_prob=args.noise_prob, gP_prob=args.gP_prob)

        if len(args.gpu_ids) > 1:
            assert args.batch_size % self.world_size == 0
            train_sampler = DistributedSampler(trainset)
            self.train_loader = DataLoader(trainset, batch_size=args.batch_size//self.world_size, num_workers=args.num_workers, sampler=train_sampler, drop_last=True)
        else:
            self.train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

        self.model = build_model(args).to(self.device)
        self.model_without_ddp = self.model
        if self.rank <= 0:
            self.logger = get_logger(args)
            if args.tb_log:
                self.tb_logger = SummaryWriter(log_dir=os.path.join(args.log_dir, args.save_dir, args.mode, 'tb_log'))
            total = sum([param.nelement() for param in self.model.parameters() if param.requires_grad])
            self.logger.info("Number of parameter: %.2fM" % (total / 1e6))
            self.logger.info(f'Model type: {args.arch}')

            if args.resume_from or args.load_from:
                pretrain = args.resume_from if args.resume_from else args.load_from
                self.logger.warning(f'Load model parameters from: {pretrain}...')
                checkpoint = torch.load(pretrain, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.logger.info('Train from scratch...')

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=True if args.drop_path_rate > 0 else False)
            self.model_without_ddp = self.model.module
        
        self.optimizer = build_optimizer(args, self.model)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        self.scheduler = build_scheduler(args, self.optimizer)
        if args.resume_from:
            checkpoint = torch.load(args.resume_from, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.args.current_epoch = checkpoint['epoch'] + 1
            self.args.current_iter = checkpoint['iter']
            if self.rank <= 0:
                self.logger.warning(f'Load optimizer and scaler...')
                self.logger.warning(f'Resume epoch to {self.args.current_epoch}, iter to {self.args.current_iter}...')

        self.criterion = torch.nn.L1Loss().to(self.device)

    def train_one_epoch(self):
        self.model.train()
        loss_per_iter = []
        load_time_list = []
        t_load = time.time()
        for batch_idx, sample in enumerate(self.train_loader):
            t_load = time.time() - t_load
            load_time_list.append(t_load)
            self.args.current_iter += 1
            if self.args.current_iter > self.args.total_iter:
                break
            
            self.scheduler.step(self.args.current_iter-1)  # update learning rate

            lq, gt = sample['lq'].to(self.device), sample['gt'].to(self.device)

            lq, gt = MoA(lq=lq, gt=gt, prob=self.args.prob,
                         probs=(self.args.mixup, self.args.cutmix, self.args.cutmixup, self.args.cutblur))

            with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # Automatic Mixed Precision
                preds = self.model(lq)
                loss = self.criterion(preds, gt)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            if self.rank <= 0:
                loss_per_iter.append(loss.detach().data.cpu())
                if self.args.tb_log:
                    self.tb_logger.add_scalar('Learning rate', self.optimizer.param_groups[0]["lr"], self.args.current_iter)
                    self.tb_logger.add_scalar('Loss/rec', loss_per_iter[-1], self.args.current_iter)

                if batch_idx % self.args.log_every_iter == 0:   # log message
                    self.logger.debug(f'epoch: {self.args.current_epoch},  '
                                      f'batch: {batch_idx+1}/{len(self.train_loader)},  '
                                      f'iter: {self.args.current_iter}/{self.args.total_iter},  '
                                      f'lr: {self.optimizer.param_groups[0]["lr"]:.4e},  '
                                      f'loss: {loss:.5f},  '
                                      f'average load time: {np.mean(load_time_list):.2f} s')
                    load_time_list = []

                if self.args.current_iter % self.args.save_every_iter == 0:  # save model
                    save_path = os.path.join(self.args.log_dir, self.args.save_dir, self.args.mode, 'checkpoint')
                    save_path = os.path.join(save_path, f'iter_{int(self.args.current_iter/1000)}k.tar')
                    torch.save({'epoch': self.args.current_epoch,
                                'iter': self.args.current_iter,
                                'model_state_dict': self.model_without_ddp.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scaler_state_dict': self.scaler.state_dict(),
                               }, save_path)
                
            t_load = time.time()

        if self.rank <= 0:
            self.logger.info(f'epoch: {self.args.current_epoch},  '
                             f'iter: {self.args.current_iter}/{self.args.total_iter},  '
                             f'lr: {self.optimizer.param_groups[0]["lr"]:.4e},  '
                             f'loss_avg: {np.mean(loss_per_iter):.5f},  ')
        self.args.current_epoch += 1

    def train(self):
        while self.args.current_iter <= self.args.total_iter:
            if self.world_size > 1:
                self.train_loader.sampler.set_epoch(self.args.current_epoch)
            self.train_one_epoch()


if __name__ == '__main__':
    args = get_args()
    set_random_seed(123 + args.crop_size)
    trainer = Trainer(args)
    trainer.train()