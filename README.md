# ZTE raw image denoising Rank 4

[竞赛官网](https://zte.hina.com/zte/denoise)

## 方案
见`docs`文件夹

## 程序说明

### 运行环境

```python
python 3.8.12
pytorch 1.8.0
cuda 11.2
```

### 训练

- 准备数据，若使用 [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) 训练集，下载 Raw-RGB images only (~20 GB)。
  
  ```
  Dataset   # 我放在了上一级目录 ../
    ├── SIDD_Medium_Raw  # SIDD数据，官网下载后解压，无需改动内部结构
    └── raw_denoising   # 比赛数据
        ├── train
              ├── ground truth
                  ├── 0_gt.dng
                  ├── ...
                  └── 99_gt.dng
              └── noisy
                  ├── 0_noise.dng
                  ├── ...
                  └── 99_noise.dng
        └── test
              ├── noisy0.dng
              ├── ...
              └── noisy9.dng
  ```
  
  放置`./docs/sidd.csv`到 `Dataset/SIDD_Medium_Raw/`下，运行`python generate_trainset.py `在单通道裁剪图像，会在`Dataset/`目录下生成相应文件夹`YOUR_DIR`，若不使用SIDD数据，注释相应代码即可，可以自行调整`generate_trainset.py`中的参数。`sidd.csv`来源见`test_SIDD.py`。

- 训练Restormer。由于方案不断改进且时间仓促，从零训练用什么参数好我也不知道，不过随便训个200-300k应该有58分。比较推荐的训练参数见`./scripts/train.sh`，这些设置参考了NAFNet文章里的训练参数，比赛时我没有用该参数训练，但我感觉用这种设置训练可以到60分。
  
  - 单卡
    
    ```
    python train.py --save_dir restormer --arch restormer --gpu_ids 0\
            --batch_size 16 --crop_size 128 --lr 2e-4 --wd 1e-4 --total_iter 200000\
            --periods 100e3,100e3 --min_lrs 2e-4,1e-6 --warmup_steps 5000\
            --dataset_dir ../Dataset/YOUR_DIR/train/ --use_checkpoint --num_workers 2
    ```
  
  - 多卡，`--nproc_per_node`设为 GPU 数量，要求 batch size 能整除 GPU 数。
    
    ```
    python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345\
            train.py --save_dir restormer --arch restormer --gpu_ids 3,4,5,7\
            --batch_size 16 --crop_size 128 --lr 2e-4 --wd 1e-4 --total_iter 200000\
            --periods 100e3,100e3 --min_lrs 2e-4,1e-6 --warmup_steps 5000\
            --dataset_dir ../Dataset/YOUR_DIR/train/ --num_workers 2
    ```
  
  `--save_dir`：训练日志和checkpoint会保存到`./log/{save_dir}/train/`。
  
  `--arch`：模型架构，另外可选择`naf`，`uformer`，`swinir`，`swinplus`。
  
  `--crop_size`：随机裁剪尺寸（4通道的尺寸），最大有效值取决于`generate_trainset.py`中的参数。
  
  `--periods`：用于cosine scheduler，`100e3,100e3`表示两个周期以及每个周期的迭代次数。
  
  `--min_lrs`：用于cosine scheduler，每个周期学习率的最小值，起始值为`--lr`。
  
  `--warmup_steps`：热身迭代数，从`--warmup_lr`线性增长到`--lr`。
  
  `--use_checkpoint`：降低显存占用，时间换空间。
  
  可以保存训练命令到 `./scripts/restormer.sh`中，然后通过`bash ./scripts/restormer.sh`进行训练。 

### 测试

- 单模+TTA
  
  ```
  python test.py --device cuda:0 --models restormer --crop_sizes 224 --strides 64\
      --weights 1 --checkpoints YOUR_CHECKPOINT_PATH --use_tta
  ```

- 多模+TTA
  
  ```
  python test.py --device cuda:0 --models restormer,uformer,naf --crop_sizes 224,384,-1 --strides 64,128,1\
      --weights 0.5,0.3,0.2 --checkpoints CKP1,CKP2,CKP3 --use_tta
  ```

`--crop_sizes`：有重叠的小块测试，使用`-1`全图测试。

`--strides`：小块测试时的步长。

测试结果会保存到`./result`。

### 验证
`val.py`使用训练时保存的checkpoints在验证集上测试效果，遍历的checkpoints范围为(start_iter, stop_iter, stride)，这三个值可以自行指定。

由于没有使用交叉验证，`val.py`已废弃。
