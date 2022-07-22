import os
import time
import torch
import rawpy
import argparse
import numpy as np
import torch.nn.functional as F
from utils.dataset import bayer_aug, ToTensor
from model.build import build_model


def get_args():
    parser = argparse.ArgumentParser('test processing')

    # TODO:
    parser.add_argument('--models', type=str, default='restormer,uformer,naf')
    parser.add_argument('--checkpoints', type=str, default='restormer.tar,uformer.tar,naf.tar')
    parser.add_argument('--weights', type=str, default='0.4,0.4,0.2')
    parser.add_argument('--use_tta', action='store_true')
    parser.add_argument('--crop_sizes', type=str, default='-1,-1,-1')
    parser.add_argument('--strides', type=str, default='64,64,64')
    parser.add_argument('--train_size', type=int, default=256, help='For NAFNet')
    parser.add_argument('--device', type=str, default='cuda:0')

    # dir setting
    parser.add_argument('--dataset_dir', type=str, default='../Dataset/raw_denoising/test/')
    parser.add_argument('--outpath', type=str, default='./result')
    
    # keep default
    parser.add_argument('--drop_path_rate', type=float, default=0)
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--mode', type=str, default='test')

    args = parser.parse_args()

    args.models = args.models.split(',')
    args.checkpoints = args.checkpoints.split(',')
    args.weights = list(map(float, args.weights.split(',')))
    assert abs(sum(args.weights) - 1) < 1e-8
    args.crop_sizes = list(map(int, args.crop_sizes.split(',')))
    args.strides = list(map(int, args.strides.split(',')))

    # manual set checkpoints (optional):
    # args.checkpoints = ['1.tar',
    #                   '2.tar',
    #                   '3.tar',
    #                   ]
    
    return args


def update_state_dict(model, pretrained_dict):
    own_state_dict = model.state_dict()
    # filter parameters
    pretrained_dict = {name: param for name, param in pretrained_dict.items() if name in own_state_dict and 'mask' not in name}
    own_state_dict.update(pretrained_dict)
    model.load_state_dict(own_state_dict)


def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
    return output_data


def write_back_dng(src_path, dest_path, raw_data):
    """
    replace dng data
    """
    width = raw_data.shape[0]
    height = raw_data.shape[1]
    falsie = os.path.getsize(src_path)
    data_len = width * height * 2
    header_len = 8

    with open(src_path, "rb") as f_in:
        data_all = f_in.read(falsie)
        dng_format = data_all[5] + data_all[6] + data_all[7]

    with open(src_path, "rb") as f_in:
        header = f_in.read(header_len)
        if dng_format != 0:
            _ = f_in.read(data_len)
            meta = f_in.read(falsie - header_len - data_len)
        else:
            meta = f_in.read(falsie - header_len - data_len)
            _ = f_in.read(data_len)

        data = raw_data.tobytes()

    with open(dest_path, "wb") as f_out:
        f_out.write(header)
        if dng_format != 0:
            f_out.write(data)
            f_out.write(meta)
        else:
            f_out.write(meta)
            f_out.write(data)

    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        print("replace raw data failed, file size mismatch!")
    else:
        print("replace raw data finished")


def write_image(input_data, height, width, tta=False):
    output_data = np.zeros((height, width), dtype=np.float64 if tta else np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[:, :, 2 * channel_y + channel_x]
    return output_data


@torch.no_grad()
def inference(lq, model, crop_size, stride):
    if crop_size == -1:
        preds = model(lq)
    
    else:
        _, _, H, W = lq.shape
        mod_pad_h = (stride - (H - crop_size) % stride) % stride
        mod_pad_w = (stride - (W - crop_size) % stride) % stride
        lq = F.pad(lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        preds = torch.zeros_like(lq)
        count = torch.zeros_like(lq)
        for h in range(0, lq.shape[2]-crop_size+1, stride):
            for w in range(0, lq.shape[3]-crop_size+1, stride):
                preds[:, :, h:h+crop_size, w:w+crop_size] += model(lq[:, :, h:h+crop_size, w:w+crop_size])
                count[:, :, h:h+crop_size, w:w+crop_size] += 1
        preds /= count
        preds = preds[:, :, :H, :W]
    
    return preds


def test(args):
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(os.path.join(args.outpath, 'data')):
        os.makedirs(os.path.join(args.outpath, 'data'))
    if not os.path.exists(os.path.join(args.outpath, 'algorithm', 'models')):
        os.makedirs(os.path.join(args.outpath, 'algorithm', 'models'))

    total_time = 0
    im_list = sorted(os.listdir(args.dataset_dir))
    for idx, file in enumerate(im_list):
        input_path = os.path.join(args.dataset_dir, file)
        output_path = os.path.join(args.outpath, 'data', 'denoise' + file[5] + '.dng')

        lq = rawpy.imread(input_path).raw_image_visible
        sample = {'0,0,0': lq}

        if args.use_tta:
            sample['0,1,0'] = bayer_aug(lq.copy(), input_pattern='BGGR', flip_h=False, flip_w=True, rot_k=0, mode='pad')
            sample['1,0,0'] = bayer_aug(lq.copy(), input_pattern='BGGR', flip_h=True, flip_w=False, rot_k=0, mode='pad')
            sample['1,1,0'] = bayer_aug(lq.copy(), input_pattern='BGGR', flip_h=True, flip_w=True, rot_k=0, mode='pad')
            sample['0,0,1'] = bayer_aug(lq.copy(), input_pattern='BGGR', flip_h=False, flip_w=False, rot_k=1, mode='pad')
            sample['0,0,3'] = bayer_aug(lq.copy(), input_pattern='BGGR', flip_h=False, flip_w=False, rot_k=3, mode='pad')

        sample = ToTensor()(sample=sample) # ['lq'][None].to(args.device)

        out_ensemble = 0.
        times = []
        for i in range(len(args.models)):
            args.arch = args.models[i]
            model = build_model(args).to(args.device)
            print(f'Load model: {args.checkpoints[i]},  {i+1}/{len(args.models)}...')
            checkpoint = torch.load(args.checkpoints[i], map_location=args.device)
            update_state_dict(model, checkpoint['model_state_dict'])
            model.eval()
            
            t = time.time()

            out_single_model = 0.
            for augs, lq_aug in sample.items():
                lq_aug = lq_aug[None].to(args.device)
                preds_aug = inference(lq_aug, model, -1 if args.arch == 'naf' else args.crop_sizes[i], args.strides[i])[0]   # [4, H'/2, W'/2]
                preds_aug = preds_aug.cpu().detach().numpy().transpose(1, 2, 0)  # [H'/2, W'/2, 4]
                preds_aug = write_image(preds_aug, preds_aug.shape[0]*2, preds_aug.shape[1]*2, tta=True)    # [H', W']

                # reverse aug
                augs = list(map(int, augs.split(',')))
                preds_aug = bayer_aug(preds_aug, input_pattern='BGGR', flip_h=bool(augs[0]), flip_w=bool(augs[1]), rot_k=(4-augs[2])%4, mode='crop')  # [H, W]
                out_single_model += preds_aug

            out_single_model = out_single_model / len(sample)
            out_ensemble += out_single_model * args.weights[i]   # [H, W], float

            times.append((time.time() - t) / 60.)
            print(f'{idx+1}/{len(im_list)}, time: {times[-1]:.2f} min')

        out_ensemble = inv_normalization(out_ensemble, black_level=1024, white_level=16383)  # [H, W], uint16
        write_back_dng(input_path, output_path, out_ensemble)

        print(f'Finished {idx+1}/{len(im_list)} image, total time: {np.sum(times):.2f} min.\n')
        total_time += np.sum(times)

    print(f'Finished test, total time: {total_time:.2f} min, average time: {total_time/len(im_list):.2f} min')
    if len(args.models) == 1:
        torch.save(model.state_dict(), os.path.join(args.outpath, 'algorithm', 'models', 'model.pth'))
    # else:
    #     for i in range(len(args.models)):
    #         checkpoint = torch.load(args.checkpoints[i], map_location=args.device)
    #         torch.save(checkpoint['model_state_dict'], os.path.join(args.outpath, 'algorithm', 'models', f'{args.models[i]}_{i}.pth'))


if __name__ == '__main__':
    args = get_args()
    test(args)