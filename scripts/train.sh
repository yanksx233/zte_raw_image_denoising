python -m torch.distributed.launch --nproc_per_node 7\
        train.py --save_dir my_model --arch my_model --gpu_ids 2,3,4,5,6,7,8\
        --batch_size 63 --crop_size 128 --prob 0.5 --cutmixup 1 --noise_prob 0.25 --gP_prob 1\
        --lr 1e-3 --wd 1e-4 --beta2 0.9 --drop_path_rate 0.15\
        --total_iter 200000 --periods 200e3 --min_lrs 1e-6 --warmup_steps 0\
        --dataset_dir ../Dataset/withsidd_score50_uint16_528x528_all/train/ --save_every_iter 5000\
        --num_workers 2