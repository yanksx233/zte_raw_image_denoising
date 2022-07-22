
python\
        train.py --save_dir restormer_more_224 --model restormer --gpu_ids 3\
        --batch_size 12 --crop_size 224 --lr 2e-4 --wd 1e-4 --total_iter 50000\
        --periods 12e4 --min_lrs 1e-6 --warmup_steps 0\
        --dataset_dir ../Dataset/withsidd_more_528x528/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1\
        --pretrain ./log/restormer_250k_sidd50_224/train/checkpoint/iter_75k.tar

python\
        train.py --save_dir restormer_more_224 --model restormer --gpu_ids 3\
        --batch_size 12 --crop_size 224 --lr 2e-4 --wd 1e-4 --total_iter 120000\
        --periods 12e4 --min_lrs 1e-6 --warmup_steps 0\
        --dataset_dir ../Dataset/comp_uint16_528x528_all/train/ --save_every_iter 5000\
        --use_checkpoint --num_workers 2 --prob 0.5 --cutmixup 1\
        --pretrain ./log/restormer_more_224/train/checkpoint/iter_50k.tar