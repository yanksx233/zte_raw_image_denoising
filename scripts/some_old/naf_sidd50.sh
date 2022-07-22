# train NAFNet
# python train.py --save_dir naf_250k_sidd50_prog --model naf --gpu_ids 1\
#         --batch_size 32 --crop_size 160 --lr 2e-4 --wd 0 --total_iter 60000\
#         --scheduler_name cosine --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
#         --dataset_dir ../Dataset/withsidd_score50_uint16_528x528_all/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 4 --prob 0.5 --cutmixup 1

# python train.py --save_dir naf_250k_sidd50_prog --model naf --gpu_ids 1\
#         --batch_size 16 --crop_size 192 --lr 2e-4 --wd 0 --total_iter 130000\
#         --scheduler_name cosine --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
#         --dataset_dir ../Dataset/withsidd_score50_uint16_528x528_all/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/naf_250k_sidd50_prog/train/checkpoint/iter_60k.tar

# python train.py --save_dir naf_250k_sidd50_prog --model naf --gpu_ids 1\
#         --batch_size 16 --crop_size 224 --lr 2e-4 --wd 0 --total_iter 190000\
#         --scheduler_name cosine --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
#         --dataset_dir ../Dataset/withsidd_score50_uint16_528x528_all/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/naf_250k_sidd50_prog/train/checkpoint/iter_130k.tar

# python train.py --save_dir naf_250k_sidd50_prog --model naf --gpu_ids 1\
#         --batch_size 16 --crop_size 256 --lr 2e-4 --wd 0 --total_iter 250000\
#         --scheduler_name cosine --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
#         --dataset_dir ../Dataset/withsidd_score50_uint16_528x528_all/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/naf_250k_sidd50_prog/train/checkpoint/iter_190k.tar

python train.py --save_dir naf_250k_sidd50_prog --model naf --gpu_ids 1\
        --batch_size 16 --crop_size 256 --lr 2e-4 --wd 0 --total_iter 210000\
        --scheduler_name cosine --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
        --dataset_dir ../Dataset/withsidd_score50_uint16_528x528_all/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/naf_250k_sidd50_prog/train/checkpoint/iter_195k.tar

python train.py --save_dir naf_250k_sidd50_prog --model naf --gpu_ids 1\
        --batch_size 16 --crop_size 256 --lr 2e-4 --wd 0 --total_iter 250000\
        --scheduler_name cosine --periods 9e4,16e4 --min_lrs 2e-4,1e-6 --warmup_steps 5000 --warmup_lr 1e-6\
        --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/naf_250k_sidd50_prog/train/checkpoint/iter_210k.tar