# train restormer
python train.py --save_dir restormer_250k_sidd43_prog --model restormer --gpu_ids 7\
        --batch_size 16 --crop_size 96 --lr 2e-4 --wd 1e-4 --total_iter 50000\
        --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000\
        --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
        --num_workers 2 --prob 0.5 --cutmixup 1

python train.py --save_dir restormer_250k_sidd43_prog --model restormer --gpu_ids 7\
        --batch_size 16 --crop_size 128 --lr 2e-4 --wd 1e-4 --total_iter 90000\
        --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000\
        --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_250k_sidd43_prog/train/checkpoint/iter_50k.tar

python train.py --save_dir restormer_250k_sidd43_prog --model restormer --gpu_ids 7\
        --batch_size 16 --crop_size 160 --lr 2e-4 --wd 1e-4 --total_iter 130000\
        --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000\
        --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_250k_sidd43_prog/train/checkpoint/iter_90k.tar

python train.py --save_dir restormer_250k_sidd43_prog --model restormer --gpu_ids 7\
        --batch_size 16 --crop_size 192 --lr 2e-4 --wd 1e-4 --total_iter 170000\
        --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000\
        --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_250k_sidd43_prog/train/checkpoint/iter_130k.tar

python train.py --save_dir restormer_250k_sidd43_prog --model restormer --gpu_ids 7\
        --batch_size 12 --crop_size 224 --lr 2e-4 --wd 1e-4 --total_iter 210000\
        --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000\
        --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_250k_sidd43_prog/train/checkpoint/iter_170k.tar

python train.py --save_dir restormer_250k_sidd43_prog --model restormer --gpu_ids 7\
        --batch_size 8 --crop_size 256 --lr 2e-4 --wd 1e-4 --total_iter 250000\
        --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000\
        --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_250k_sidd43_prog/train/checkpoint/iter_210k.tar

python train.py --save_dir restormer_250k_sidd2comp_prog --model restormer --gpu_ids 7\
        --batch_size 8 --crop_size 256 --lr 2e-4 --wd 1e-4 --total_iter 250000\
        --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000\
        --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_250k_sidd43_prog/train/checkpoint/iter_210k.tar