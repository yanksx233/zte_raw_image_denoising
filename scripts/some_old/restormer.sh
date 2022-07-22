#-m torch.distributed.launch --nproc_per_node 3\
# python\
#         train.py --save_dir restormer_250k_sidd50_224 --model restormer --gpu_ids 1\
#         --batch_size 12 --crop_size 224 --lr 2e-4 --wd 1e-4 --total_iter 40000\
#         --periods 6e4 --min_lrs 1e-6 --warmup_steps 0\
#         --dataset_dir ../Dataset/withsidd_score50_uint16_528x528_all/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_250k_sidd43_prog/train/checkpoint/iter_250k.tar

# python\
#         train.py --save_dir restormer_250k_sidd50_224 --model restormer --gpu_ids 1\
#         --batch_size 12 --crop_size 224 --lr 2e-4 --wd 1e-4 --total_iter 60000\
#         --periods 6e4 --min_lrs 1e-6 --warmup_steps 0\
#         --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_250k_sidd50_224/train/checkpoint/iter_40k.tar

# comment load optimizer
# python\
#         train.py --save_dir restormer_pseudo1_224 --model restormer --gpu_ids 1\
#         --batch_size 12 --crop_size 224 --lr 2e-4 --wd 1e-4 --total_iter 40000\
#         --periods 8e4 --min_lrs 1e-6 --warmup_steps 0\
#         --dataset_dir ../Dataset/score50_pseudo1/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_250k_sidd50_224/train/checkpoint/iter_60k.tar

# python\
#         train.py --save_dir restormer_pseudo1_224 --model restormer --gpu_ids 1\
#         --batch_size 12 --crop_size 224 --lr 2e-4 --wd 1e-4 --total_iter 80000\
#         --periods 8e4 --min_lrs 1e-6 --warmup_steps 0\
#         --dataset_dir ../Dataset/comp_pseudo1/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 2 --prob 0.8 --cutmixup 1 --pretrain ./log/restormer_pseudo1_224/train/checkpoint/iter_40k.tar

# python\
#         train.py --save_dir restormer_syn_224 --model restormer --gpu_ids 1\
#         --batch_size 12 --crop_size 224 --lr 2e-4 --wd 1e-4 --total_iter 30000\
#         --periods 15e4 --min_lrs 1e-6 --warmup_steps 0\
#         --dataset_dir ../Dataset/withsidd_score50_uint16_528x528_all/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_250k_sidd50_224/train/checkpoint/iter_75k.tar

# python\
#         train.py --save_dir restormer_syn_224 --model restormer --gpu_ids 1\
#         --batch_size 12 --crop_size 224 --lr 2e-4 --wd 1e-4 --total_iter 150000\
#         --periods 15e4 --min_lrs 1e-6 --warmup_steps 0\
#         --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_syn_224/train/checkpoint/iter_30k.tar

# python -m torch.distributed.launch --nproc_per_node 4\
#         train.py --save_dir restormer_syn_comp_224 --model restormer --gpu_ids 3,4,5,7\
#         --batch_size 12 --crop_size 224 --lr 2e-4 --wd 1e-4 --total_iter 200000\
#         --periods 200e3 --min_lrs 1e-6 --warmup_steps 0\
#         --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
#         --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_250k_sidd50_224/train/checkpoint/iter_75k.tar

# python -m torch.distributed.launch --nproc_per_node 4\
#         train.py --save_dir restormer_syn_comp_224 --model restormer --gpu_ids 3,4,5,7\
#         --batch_size 12 --crop_size 224 --lr 2e-4 --wd 1e-4 --total_iter 70000\
#         --periods 200e3 --min_lrs 1e-6 --warmup_steps 0\
#         --dataset_dir ../Dataset/withsidd_score50_uint16_528x528_all/train/ --save_every_iter 5000\
#         --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_syn_comp_224/train/checkpoint/iter_30k.tar

# python -m torch.distributed.launch --nproc_per_node 4\
#         train.py --save_dir restormer_syn_comp_224 --model restormer --gpu_ids 3,4,5,7\
#         --batch_size 12 --crop_size 224 --lr 2e-4 --wd 1e-4 --total_iter 200000\
#         --periods 200e3 --min_lrs 1e-6 --warmup_steps 0\
#         --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
#         --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_syn_comp_224/train/checkpoint/iter_70k.tar

# python\
#         train.py --save_dir restormer_syn_comp_224 --model restormer --gpu_ids 6 --noise_model g\
#         --batch_size 12 --crop_size 160 --lr 2e-4 --wd 1e-4 --total_iter 200000\
#         --periods 200e3 --min_lrs 1e-6 --warmup_steps 0\
#         --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
#         --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_syn_comp_224/train/checkpoint/iter_40k.tar



# python -m torch.distributed.launch --nproc_per_node 4\
#         train.py --save_dir restormer_syn_gp_comp_192 --model restormer --gpu_ids 3,4,5,7\
#         --batch_size 12 --crop_size 192 --lr 2e-4 --wd 1e-4 --total_iter 70000\
#         --periods 200e3 --min_lrs 1e-5 --warmup_steps 0\
#         --dataset_dir ../Dataset/withsidd_score50_uint16_528x528_all/train/ --save_every_iter 5000\
#         --num_workers 1 --prob 0.5 --cutmixup 1 --pretrain ./log/restormer_syn_gp_comp_192/train/checkpoint/iter_20k.tar

python -m torch.distributed.launch --nproc_per_node 4\
        train.py --save_dir restormer_syn_gp_comp_192 --model restormer --gpu_ids 3,4,5,7\
        --batch_size 12 --crop_size 192 --lr 2e-4 --wd 1e-4 --total_iter 200000\
        --periods 200e3 --min_lrs 7e-5 --warmup_steps 0\
        --dataset_dir ../Dataset/comp_pse/train/ --save_every_iter 5000\
        --num_workers 1 --prob 1 --cutmixup 1 --noise_prob 0.3 --gP_prob 1\
        --pretrain ./log/restormer_syn_gp_comp_192/train/checkpoint/iter_165k.tar