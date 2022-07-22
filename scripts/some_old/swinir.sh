# train swinirir
# python train.py --save_dir swinir_150k_comp --model swinir --gpu_ids 8\
#         --batch_size 10 --crop_size 64 --lr 2e-4 --wd 1e-4 --total_iter 10000\
#         --scheduler_name cosine --periods 8e4,7e4 --decay_step 50000 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
#         --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
#         --num_workers 2 --prob 0

# python train.py --save_dir swinir_150k_comp --model swinir --gpu_ids 8\
#         --batch_size 10 --crop_size 64 --lr 2e-4 --wd 1e-4 --total_iter 150000\
#         --scheduler_name cosine --periods 8e4,7e4 --decay_step 50000 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
#         --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
#         --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/swinir_150k_comp/train/checkpoint/iter_10k.tar

python train.py --save_dir swinir_250k_sidd43 --model swinir --gpu_ids 8\
        --batch_size 16 --crop_size 64 --lr 2e-4 --wd 1e-4 --total_iter 200000\
        --scheduler_name cosine --periods 9e4,16e4 --decay_step 50000 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
        --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0

python train.py --save_dir swinir_250k_sidd43 --model swinir --gpu_ids 8\
        --batch_size 16 --crop_size 96 --lr 2e-4 --wd 1e-4 --total_iter 250000\
        --scheduler_name cosine --periods 9e4,16e4 --decay_step 50000 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
        --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0 --pretrain ./log/swinir_250k_sidd43/train/checkpoint/iter_200k.tar

# python train.py --save_dir swinir_250k_sidd43 --model swinir --gpu_ids 8\
#         --batch_size 10 --crop_size 64 --lr 2e-4 --wd 1e-4 --total_iter 250000\
#         --scheduler_name cosine --periods 9e4,16e4 --decay_step 50000 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
#         --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
#         --num_workers 2 --prob 0 --cutmixup 1 --pretrain ./log/swinir_250k_sidd43/train/checkpoint/iter_10k.tar


# python train.py --save_dir swinir_150k_comp --model swinir --gpu_ids 8\
#         --batch_size 10 --crop_size 96 --lr 2e-4 --wd 1e-4 --total_iter 100000\
#         --scheduler_name step --decay_step 50000 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
#         --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 2 --prob 0.5 --mixup 1 --pretrain ./log/swinir_150k_comp/train/checkpoint/iter_50k.tar

# python train.py --save_dir swinir_150k_comp --model swinir --gpu_ids 8\
#         --batch_size 10 --crop_size 128 --lr 2e-4 --wd 1e-4 --total_iter 150000\
#         --scheduler_name step --decay_step 50000 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
#         --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 2 --prob 0.5 --mixup 1 --pretrain ./log/swinir_150k_comp/train/checkpoint/iter_100k.tar