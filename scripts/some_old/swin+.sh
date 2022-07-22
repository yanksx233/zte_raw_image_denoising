# train swinplusir
# python train.py --save_dir swinplus_250k_sidd43 --model swinplus --gpu_ids 8\
#         --batch_size 16 --crop_size 64 --lr 2e-4 --wd 1e-4 --total_iter 30000\
#         --scheduler_name cosine --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
#         --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 2 --prob 0

# python -m torch.distributed.launch --nproc_per_node 4\
#         train.py --save_dir swinplus_250k_sidd43 --model swinplus --gpu_ids 0,1,2,3\
#         --batch_size 16 --crop_size 64 --lr 2e-4 --wd 1e-4 --total_iter 200000\
#         --scheduler_name cosine --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
#         --dataset_dir ../Dataset/withsidd_psnr43_uint16_528x528_all/train/ --save_every_iter 5000\
#         --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/swinplus_250k_sidd43/train/checkpoint/iter_130k.tar

python -m torch.distributed.launch --nproc_per_node 4\
        train.py --save_dir swinplus_250k_sidd43 --model swinplus --gpu_ids 4,0,2,5\
        --batch_size 16 --crop_size 80 --lr 2e-4 --wd 1e-4 --total_iter 250000\
        --scheduler_name cosine --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
        --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
        --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/swinplus_250k_sidd43/train/checkpoint/iter_200k.tar