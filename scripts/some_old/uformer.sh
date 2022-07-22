# train uformer
# python train.py --save_dir uformer_250k_sidd43_prog --model uformer --gpu_ids 4\
#         --batch_size 16 --crop_size 128 --lr 2e-4 --wd 1e-4 --total_iter 15000\
#         --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000\
#         --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
#         --num_workers 2 --prob 0

python train.py --save_dir uformer_250k_sidd43_prog --model uformer --gpu_ids 4\
        --batch_size 16 --crop_size 128 --lr 2e-4 --wd 1e-4 --total_iter 200000\
        --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000\
        --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
        --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/uformer_250k_sidd43_prog/train/checkpoint/iter_60k.tar

python train.py --save_dir uformer_250k_sidd43_prog --model uformer --gpu_ids 4\
        --batch_size 16 --crop_size 256 --lr 2e-4 --wd 1e-4 --total_iter 220000\
        --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000\
        --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/uformer_250k_sidd43_prog/train/checkpoint/iter_200k.tar

python train.py --save_dir uformer_250k_sidd43_prog --model uformer --gpu_ids 8\
        --batch_size 16 --crop_size 256 --lr 2e-4 --wd 1e-4 --total_iter 250000\
        --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000\
        --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/uformer_250k_sidd43_prog/train/checkpoint/iter_220k.tar