import os
import time
import torch
import rawpy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import h5py
import matplotlib.pyplot as plt
from .noise import NoiseModel


BAYER_PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"]
NORMALIZATION_MODE = ["crop", "pad"]


def bayer_unify(raw: np.ndarray, input_pattern: str, target_pattern: str, mode: str) -> np.ndarray:
    """
    Convert a bayer raw image from one bayer pattern to another.
    Parameters
    ----------
    raw : np.ndarray in shape (H, W)
        Bayer raw image to be unified.
    input_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        The bayer pattern of the input image.
    target_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        The expected output pattern.
    mode: {"crop", "pad"}
        The way to handle submosaic shift. "crop" abandons the outmost pixels,
        and "pad" introduces extra pixels. Use "crop" in training and "pad" in
        testing.
    """
    if input_pattern not in BAYER_PATTERNS:
        raise ValueError('Unknown input bayer pattern!')
    if target_pattern not in BAYER_PATTERNS:
        raise ValueError('Unknown target bayer pattern!')
    if mode not in NORMALIZATION_MODE:
        raise ValueError('Unknown normalization mode!')
    if not isinstance(raw, np.ndarray) or len(raw.shape) != 2:
        raise ValueError('raw should be a 2-dimensional numpy.ndarray!')

    if input_pattern == target_pattern:
        h_offset, w_offset = 0, 0
    elif input_pattern[0] == target_pattern[2] and input_pattern[1] == target_pattern[3]:
        h_offset, w_offset = 1, 0
    elif input_pattern[0] == target_pattern[1] and input_pattern[2] == target_pattern[3]:
        h_offset, w_offset = 0, 1
    elif input_pattern[0] == target_pattern[3] and input_pattern[1] == target_pattern[2]:
        h_offset, w_offset = 1, 1
    else:  # This is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]
        raise RuntimeError('Unexpected pair of input and target bayer pattern!')

    if mode == "pad":
        out = np.pad(raw, [[h_offset, h_offset], [w_offset, w_offset]], 'reflect')
    elif mode == "crop":
        h, w = raw.shape
        out = raw[h_offset:h - h_offset, w_offset:w - w_offset]
    else:
        raise ValueError('Unknown normalization mode!')

    return out


def bayer_aug(raw: np.ndarray, input_pattern: str, flip_h=False, flip_w=False, transpose=False, rot_k=0, mode='crop') -> np.ndarray:
    """
    Apply augmentation to a bayer raw image.
    Parameters
    ----------
    raw : np.ndarray in shape (H, W)
        Bayer raw image to be augmented. H and W must be even numbers.
    input_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        The bayer pattern of the input image.
    flip_h : bool
        If True, do vertical flip.
    flip_w : bool
        If True, do horizontal flip.
    transpose : bool
        If True, do transpose.
    """

    if input_pattern not in BAYER_PATTERNS:
        raise ValueError('Unknown input bayer pattern!')
    if not isinstance(raw, np.ndarray) or len(raw.shape) != 2:
        raise ValueError('raw should be a 2-dimensional numpy.ndarray')
    if raw.shape[0] % 2 == 1 or raw.shape[1] % 2 == 1:
        raise ValueError('raw should have even number of height and width!')

    aug_pattern, target_pattern = input_pattern, input_pattern

    out = raw
    if flip_h:
        out = out[::-1, :]
        aug_pattern = aug_pattern[2] + aug_pattern[3] + aug_pattern[0] + aug_pattern[1]
    if flip_w:
        out = out[:, ::-1]
        aug_pattern = aug_pattern[1] + aug_pattern[0] + aug_pattern[3] + aug_pattern[2]
    if transpose:
        out = out.T
        aug_pattern = aug_pattern[0] + aug_pattern[2] + aug_pattern[1] + aug_pattern[3]
    if rot_k > 0:
        out = np.rot90(out, rot_k)
        if rot_k == 1:
            aug_pattern = aug_pattern[1] + aug_pattern[3] + aug_pattern[0] + aug_pattern[2]
        elif rot_k == 2:
            aug_pattern = aug_pattern[3] + aug_pattern[2] + aug_pattern[1] + aug_pattern[0]
        elif rot_k == 3:
            aug_pattern = aug_pattern[2] + aug_pattern[0] + aug_pattern[3] + aug_pattern[1]
        else:
            raise ValueError('Rotation error!')

    out = bayer_unify(out, aug_pattern, target_pattern, mode)
    return out


def geometry_aug(sample, input_pattern):
    flip_h = bool(np.random.randint(2))
    flip_w = bool(np.random.randint(2))
    rot_k = np.random.randint(0, 4)

    sample['lq'] = bayer_aug(sample['lq'], flip_h=flip_h, flip_w=flip_w, rot_k=rot_k, input_pattern=input_pattern)
    sample['gt'] = bayer_aug(sample['gt'], flip_h=flip_h, flip_w=flip_w, rot_k=rot_k, input_pattern=input_pattern)
    return sample


def mixup(lq, gt, prob=0.5, alpha=0.6):
    # lq (tensor): [B, C, H, W]
    # gt (tensor): [B, C, H, W]

    if np.random.rand(1) >= prob:
        return lq, gt

    B = gt.shape[0]
    rand_indices = torch.randperm(B)
    lam = torch.distributions.beta.Beta(alpha, alpha).rsample((B, 1, 1, 1)).to(gt.device)

    lq = lam * lq + (1 - lam) * lq[rand_indices]
    gt = lam * gt + (1 - lam) * gt[rand_indices]

    return lq, gt


def cutmix(lq, gt, prob=0.5, max_ratio=1.0):
    # lq (tensor): [B, C, H, W]
    # gt (tensor): [B, C, H, W]

    if max_ratio <= 0 or np.random.rand(1) >= prob:
        return lq, gt

    B, _, H, W = lq.shape
    rand_indices = torch.randperm(B)

    cut_h_size = np.random.randint(0, int(H*max_ratio))
    cut_w_size = np.random.randint(0, int(W*max_ratio))

    h0 = np.random.randint(0, H-cut_h_size+1)
    w0 = np.random.randint(0, W-cut_w_size+1)

    lq[:, :, h0:h0+cut_h_size, w0:w0+cut_w_size] = lq[rand_indices, :, h0:h0+cut_h_size, w0:w0+cut_w_size]
    gt[:, :, h0:h0+cut_h_size, w0:w0+cut_w_size] = gt[rand_indices, :, h0:h0+cut_h_size, w0:w0+cut_w_size]

    return lq, gt


def cutmixup(lq, gt, prob=0.5, max_ratio=1.0, alpha=0.6):
    # lq (tensor): [B, C, H, W]
    # gt (tensor): [B, C, H, W]

    if max_ratio <= 0 or np.random.rand(1) >= prob:
        return lq, gt

    lq_mix, gt_mix = mixup(lq.clone(), gt.clone(), prob=1, alpha=alpha)
    
    B, _, H, W = lq.shape
    cut_h_size = np.random.randint(0, int(H*max_ratio))
    cut_w_size = np.random.randint(0, int(W*max_ratio))

    h0 = np.random.randint(0, H-cut_h_size+1)
    w0 = np.random.randint(0, W-cut_w_size+1)
    h1 = np.random.randint(0, H-cut_h_size+1)
    w1 = np.random.randint(0, W-cut_w_size+1)

    # inside or outside
    if np.random.random() > 0.5:
        lq[:, :, h0:h0+cut_h_size, w0:w0+cut_w_size] = lq_mix[:, :, h1:h1+cut_h_size, w1:w1+cut_w_size]
        gt[:, :, h0:h0+cut_h_size, w0:w0+cut_w_size] = gt_mix[:, :, h1:h1+cut_h_size, w1:w1+cut_w_size]

    else:
        lq_mix[:, :, h0:h0+cut_h_size, w0:w0+cut_w_size] = lq[:, :, h1:h1+cut_h_size, w1:w1+cut_w_size]
        gt_mix[:, :, h0:h0+cut_h_size, w0:w0+cut_w_size] = gt[:, :, h1:h1+cut_h_size, w1:w1+cut_w_size]
        lq, gt = lq_mix, gt_mix

    return lq, gt


def cutblur(lq, gt, prob=0.5, alpha=0.7):
    # lq (tensor): [B, C, H, W]
    # gt (tensor): [B, C, H, W]

    if alpha <= 0 or np.random.rand(1) >= prob:
        return lq, gt

    H, W = gt.shape[-2:]
    cut_ratio = np.random.randn() * 0.01 + alpha
    cut_ratio = np.clip(cut_ratio, 0, 1)

    cut_h_size = np.random.randint(0, int(H*cut_ratio))
    cut_w_size = np.random.randint(0, int(W*cut_ratio))

    h0 = np.random.randint(0, H-cut_h_size+1)
    w0 = np.random.randint(0, W-cut_w_size+1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        lq[:, :, h0:h0+cut_h_size, w0:w0+cut_w_size] = gt[:, :, h0:h0+cut_h_size, w0:w0+cut_w_size]
    else:
        lq_aug = gt.clone()
        lq_aug[:, :, h0:h0+cut_h_size, w0:w0+cut_w_size] = lq[:, :, h0:h0+cut_h_size, w0:w0+cut_w_size]
        lq = lq_aug

    return lq, gt


def MoA(lq, gt, prob=0.8, 
        augs=(mixup, cutmix, cutmixup, cutblur), 
        probs=(0.1, 0.3, 0.3, 0.3),
        ):
# Mixture of Augmentations 
    if np.random.rand(1) >= prob:
        return lq, gt
    
    assert np.sum(probs) == 1
    idx = np.random.choice(len(augs), p=probs)
    lq, gt = augs[idx](lq=lq, gt=gt, prob=1)
    return lq, gt


def to_4channels(raw):
    # input (np.ndarray): [H, W]
    # return (np.ndarray): [H/2, W/2, 4]

    height = raw.shape[0]
    width = raw.shape[1]

    raw_data_expand = np.expand_dims(raw, axis=2)  # [H, W, 1]
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)  # [H/2, W/2, 4]
    return raw_data_expand_c


class ToTensor(object):
    def __call__(self, sample, norm=True):
        # sample (dict: np.ndarray): shape of [H, W]
        # output (dict: torch.tensor): range of [0, 1], shape of [4, H/2, W/2]

        for k in sample:
            sample[k] = to_4channels(sample[k])
            sample[k] = torch.Tensor((sample[k].astype(float) - 1024) / (16383 - 1024)) if norm else torch.Tensor(sample[k].astype(float))
            sample[k] = sample[k].permute(2, 0, 1).clone()
        return sample


class TrainSet(Dataset):
    def __init__(self, data_dir, crop_size=64, pattern='BGGR', train=True, noise_prob=0.5, gP_prob=1.):
        self.data_dir = data_dir
        self.crop_size = crop_size * 2  # 4-channels size to 1-channel size
        self.pattern = pattern
        self.train = train
        self.gP_maker = NoiseModel(model='g+P') if train else None
        self.g_maker = NoiseModel(model='g') if train else None
        self.noise_prob = noise_prob
        self.gP_prob = gP_prob

        self.im_list = os.listdir(os.path.join(data_dir, 'noisy'))
        self.len = len(self.im_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file = self.im_list[idx]
        lq = np.load(os.path.join(self.data_dir, 'noisy', file))
        gt = np.load(os.path.join(self.data_dir, 'gt', file))

        sample = {'lq': lq, 'gt': gt}
        if self.train:
            sample = geometry_aug(sample, self.pattern)
            # crop
            H, W = sample['gt'].shape
            h0 = np.random.randint(0, H-self.crop_size+1) // 4 * 4
            w0 = np.random.randint(0, W-self.crop_size+1) // 4 * 4

            for k in sample:
                sample[k] = sample[k][h0:h0+self.crop_size, w0:w0+self.crop_size]
                sample[k] = to_4channels(sample[k])
                sample[k] = (sample[k].astype(float) - 1024) / (16383 - 1024)
            
            # add noise
            if np.random.rand() < self.noise_prob:
                if np.random.rand() < self.gP_prob:
                    sample['lq'] = self.gP_maker(np.clip(sample['gt'], 0., 1.))
                    sample['lq'] = np.clip(sample['lq'], 0., 1.)
                else:
                    sample['lq'] = self.g_maker(np.clip(sample['gt'], 0., 1.))
                    sample['lq'] = np.clip(sample['lq'], 0., 1.)

            for k in sample:
                sample[k] = torch.Tensor(sample[k]).permute(2, 0, 1).clone()

        return sample


class SIDD(Dataset):
    _pattern_table = {'GP': 'BGGR',
                      'IP': 'RGGB', 
                      'S6': 'GRBG', 
                      'N6': 'BGGR', 
                      'G4': 'BGGR',
                     }

    def __init__(self, data_dir, target_pattern='BGGR'):
        self.data_dir = data_dir
        self.target_pattern = target_pattern
        self.imlist = os.listdir(data_dir)
        self.len = len(self.imlist) * 2

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        dir_idx = idx // 2
        idx = idx % 2
        dir = self.imlist[dir_idx]
        lq = h5py.File(os.path.join(self.data_dir, dir, f'{dir[:4]}_NOISY_RAW_01{idx}.MAT'))
        gt = h5py.File(os.path.join(self.data_dir, dir, f'{dir[:4]}_GT_RAW_01{idx}.MAT'))
        input_pattern = self.get_input_pattern(dir)
        lq = bayer_unify(np.transpose(lq['x']), input_pattern=input_pattern, target_pattern=self.target_pattern, mode='crop')
        gt = bayer_unify(np.transpose(gt['x']), input_pattern=input_pattern, target_pattern=self.target_pattern, mode='crop')

        sample = {'lq': lq, 'gt': gt}
        return sample, dir

    def get_input_pattern(self, dir_name):
        for k, v in self._pattern_table.items():
            if k in dir_name:
                return v
        raise ValueError(f'Unknown Bayer pattern for {dir_name}')


class SIDDFromCSV(SIDD):
    def __init__(self, data_dir, index='score', threshold=50, target_pattern='BGGR'):
        self.data_dir = os.path.join(data_dir, 'Data')
        self.df = pd.read_csv(os.path.join(data_dir, 'sidd.csv'))
        self.df = self.df[self.df[index] > threshold]
        self.len = len(self.df)
        self.target_pattern = target_pattern

    def __getitem__(self, idx):
        dir = self.df.iloc[idx]['dir']
        file_idx = self.df.iloc[idx]['idx']
        lq = h5py.File(os.path.join(self.data_dir, dir, f'{dir[:4]}_NOISY_RAW_01{file_idx}.MAT'))
        gt = h5py.File(os.path.join(self.data_dir, dir, f'{dir[:4]}_GT_RAW_01{file_idx}.MAT'))
        input_pattern = self.get_input_pattern(dir)
        lq = bayer_unify(np.transpose(lq['x']), input_pattern=input_pattern, target_pattern=self.target_pattern, mode='crop')
        gt = bayer_unify(np.transpose(gt['x']), input_pattern=input_pattern, target_pattern=self.target_pattern, mode='crop')

        sample = {'lq': lq, 'gt': gt}
        return sample


if __name__ == '__main__':
    im = np.arange(400).reshape(20, 20)
    rot_k = 0
    im_aug_pad = bayer_aug(im, input_pattern='BGGR', flip_h=True, flip_w=True, rot_k=rot_k, mode='pad')
    im_rev_aug = bayer_aug(im_aug_pad, input_pattern='BGGR', flip_h=True, flip_w=True, rot_k=(4-rot_k)%4, mode='crop')
    print('inp', im, sep='\n')
    print('aug', im_aug_pad, sep='\n')
    print('re_aug', im_rev_aug, sep='\n')
    print(im == im_rev_aug)

