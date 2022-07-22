import math
import torch
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


def cal_psnr_and_ssim(img1, img2):
    # input: pixel range [0, 1], tensor type
    img1_np = np.array(torch.squeeze(img1.data.cpu(), 0).permute(1, 2, 0))
    img2_np = np.array(torch.squeeze(img2.data.cpu(), 0).permute(1, 2, 0))

    return psnr(img1_np, img2_np), ssim(img1_np, img2_np, multichannel=True, data_range=1)


def build_optimizer(args, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip_keywords = []
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords.append(model.no_weight_decay_keywords())
    parameters = set_weight_decay(model, skip_keywords)
    optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.wd, betas=(args.beta1, args.beta2))
    return optimizer


def set_weight_decay(model, skip_keywords):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue    # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if name.endswith(keyword):
            isin = True
    return isin


def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.
    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7
    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.
    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        warmup_lr_init (folat): The initial learning rate for warm up.
        warmup_t (int): The iteration of warm up.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=(1, ),
                 eta_mins=(0, ),
                 warmup_lr_init=0,
                 warmup_t=0,
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        self.warmup_lr_init = warmup_lr_init
        self.warmup_t = warmup_t
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_t:
            return [
            self.warmup_lr_init + (base_lr - self.warmup_lr_init) * self.last_epoch / self.warmup_t
            for base_lr in self.base_lrs
        ]

        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


def build_scheduler(args, optimizer):
    lr_scheduler = None
    if args.scheduler_name == 'cosine':
        lr_scheduler = CosineAnnealingRestartCyclicLR(
            optimizer,
            periods=args.periods,
            restart_weights=(1,)*len(args.periods),
            eta_mins=args.min_lrs,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_steps
        )

    elif args.scheduler_name == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_step,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_steps,
            t_in_epochs=True,
        )
    else:
        raise SystemExit(f'Undefined scheduler: {args.scheduler_name}!')

    return lr_scheduler