import os
import rawpy
import random
import numpy as np
from utils.dataset import SIDD, SIDDFromCSV

if __name__ == '__main__':
    random.seed(233)
    np.random.seed(233)
    fold = 0  # or 1
    out_size = 528
    stride = 128
    val_stride = out_size
    index = 'score'
    threshold = 50
    data_dir = '../Dataset/raw_denoising/train'  # dir of competition data
    sidd_dir = '../Dataset/SIDD_Medium_Raw'

    save_dir_name = f'withsidd_{index}{threshold}_uint16_{out_size}x{out_size}_all'

    os.makedirs(f'../Dataset/{save_dir_name}/train/noisy/')
    os.makedirs(f'../Dataset/{save_dir_name}/train/gt/')
    os.makedirs(f'../Dataset/{save_dir_name}/val/noisy/')
    os.makedirs(f'../Dataset/{save_dir_name}/val/gt/')
    
    lq_list = sorted(os.listdir(os.path.join(data_dir, 'noisy')))
    gt_list = sorted(os.listdir(os.path.join(data_dir, 'ground truth')))
    assert len(lq_list) == len(gt_list)

    # val_indices = np.random.choice(len(lq_list), size=10, replace=False)[(fold*5):(fold*5)+5]
    val_indices = []
    print(val_indices)

    train_count = 0
    val_count = 0
    for idx in range(len(lq_list)):
        lq = rawpy.imread(os.path.join(data_dir, 'noisy', lq_list[idx])).raw_image_visible
        gt = rawpy.imread(os.path.join(data_dir, 'ground truth', gt_list[idx])).raw_image_visible

        H, W = lq.shape
        print(lq_list[idx], gt_list[idx])

        if idx in val_indices:
            for h0 in range(0, H-out_size+1, val_stride):
                for w0 in range(0, W-out_size+1, val_stride):
                    lq_patch = lq[h0:h0+out_size, w0:w0+out_size]
                    gt_patch = gt[h0:h0+out_size, w0:w0+out_size]       
                    np.save(f'../Dataset/{save_dir_name}/val/noisy/{val_count}.npy', lq_patch)
                    np.save(f'../Dataset/{save_dir_name}/val/gt/{val_count}.npy', gt_patch)
                    val_count += 1
            
        else:
            for h0 in range(0, H-out_size+1, stride):
                for w0 in range(0, W-out_size+1, stride):
                    lq_patch = lq[h0:h0+out_size, w0:w0+out_size]
                    gt_patch = gt[h0:h0+out_size, w0:w0+out_size]      
                    np.save(f'../Dataset/{save_dir_name}/train/noisy/{train_count}.npy', lq_patch)
                    np.save(f'../Dataset/{save_dir_name}/train/gt/{train_count}.npy', gt_patch)
                    train_count += 1


    sidd_dataset = SIDDFromCSV(sidd_dir, index=index, threshold=threshold)
    # sidd_dataset = SIDD(sidd_dir + '/Data')  ##  use this if .csv file is not existed

    for i, sample in enumerate(iter(sidd_dataset)):
    # for i, (sample, _) in enumerate(iter(sidd_dataset)):  ##  use this if .csv file is not existed
        lq = sample['lq']
        gt = sample['gt']
        
        print(f'SIDD: {i+1}/{len(sidd_dataset)}')

        lq = (lq * (16383 - 1024) + 1024).astype(np.uint16)
        gt = (gt * (16383 - 1024) + 1024).astype(np.uint16)

        H, W = lq.shape
        for h0 in range(0, H-out_size+1, stride):
            for w0 in range(0, W-out_size+1, stride):
                lq_patch = lq[h0:h0+out_size, w0:w0+out_size]
                gt_patch = gt[h0:h0+out_size, w0:w0+out_size]

                np.save(f'../Dataset/{save_dir_name}/train/noisy/{train_count}.npy', lq_patch)
                np.save(f'../Dataset/{save_dir_name}/train/gt/{train_count}.npy', gt_patch)
                train_count += 1