# ERRNet DIP26 Guide

## 1. Clone and Download

### 1.1 Clone the repository

```bash
git clone https://github.com/innerway-xq/ERRNet
cd ERRNet
git checkout dip26
```

### 1.2 Setup the environment

```bash
conda create -n errnet python=3.10 -y
conda activate errnet
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -U pip wheel "setuptools<82"
pip install visdom==0.2.4 --no-build-isolation
```

Notes:
- Install the correct `torch`/`torchvision` version for your machine. Use a CUDA build if you have a GPU, and a CPU build otherwise. (Refer to https://pytorch.org/get-started/previous-versions/)

### 1.3 Download files

Download through [BaiduYun](https://pan.baidu.com/s/1MWb4eT18ySjogKVlcfPozg?pwd=egv2) or [GoogleDrive](https://drive.google.com/drive/folders/1_tN6JDlAmKZTgaqniQep1YJXmbFwGav7?usp=drive_link), then unzip and place files in ERRNet like this: 
```text
ERRNet/
  checkpoints/
    errnet/
      errnet_060_00463920.pt
  datasets/
    raw_data/
      VOCdevkit/
      CEILNet/
      real89/
      robustsirr_test_dataset/
      Dataset/
```

## 2. Prepare the Training and Testing Data


Run:

```bash
python datasets/prepare_test_data.py
python datasets/prepare_train_data.py
```


## 3. Testing

The testing script is `test_errnet.py`.

### 3.1 Benchmark testing

Supported benchmark names:

- `ceilnet_table2`
- `real20`
- `postcard`
- `objects`
- `wild`
- `sir2_withgt`

```bash
# gpu
python test_errnet.py --name errnet --dataset [dataset] -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
# cpu
python test_errnet.py --name errnet_cpu --dataset [dataset] -r --gpu_ids -1 --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
```


### 3.2 Test on your own images

If you only want to run the model on your own reflection images, ground truth is not required. Put your images in any folder, for example:

```text
datasets/raw_data/my_test_images/
  img1.jpg
  img2.jpg
```

Run:
```bash
python test_errnet.py --name errnet --dataset custom --input_dir ./datasets/raw_data/my_test_images -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
```

Each image will have its own subfolder. You will usually see:

- `m_input.png`: the input image
- `errnet.png` or `errnet_cpu.png`: the model output

## 4. Training

There are two training stages:

- Aligned-data training: `train_errnet.py`
- Unaligned-data finetuning: `train_errnet_unaligned.py`

### 4.1 Train the aligned baseline

```bash
# gpu
python train_errnet.py --name errnet --hyper
# cpu
python train_errnet.py --name errnet_cpu --hyper --gpu_ids -1
```

### 4.2 Finetune on unaligned data


#### GPU version

```bash
# gpu
python train_errnet_unaligned.py --name errnet_unaligned_ft --hyper -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --unaligned_loss vgg
# cpu
python train_errnet_unaligned.py --name errnet_unaligned_ft_cpu --hyper -r --gpu_ids -1 --icnn_path checkpoints/errnet/errnet_060_00463920.pt --unaligned_loss vgg
```


## 5. Baseline Result

> checkpoints/errnet/errnet_060_00463920.pt

| Dataset | PSNR | SSIM | NCC | LMSE |
| --- | --- | --- | --- | --- |
| CEILNet| 27.88 | 0.9407 | 0.9808 | 0.0048 |
| real20 | 23.55 | 0.8285 | 0.8877 | 0.0201 |
| objects | 24.85 | 0.8980 | 0.9817 | 0.0029|
| postcard | 22.07 | 0.8773 | 0.9463 | 0.0044 |
| wild | 25.18 | 0.886 | 0.9359 | 0.0083|

| Dataset | PSNR | SSIM | NCC | LMSE |
| --- | --- | --- | --- | --- |
| CEILNet | 27.22 | 0.9383 | 0.9754 | 0.0047 | down
| real20 | 24.81 | 0.8439 | 0.9141 | 0.0168 | up
| objects | 24.86 | 0.8985 | 0.9799 | 0.0034 | draw
| postcard | 20.78 | 0.8757 | 0.9233 | 0.0044 | down
| wild | 25.76 | 0.9070 | 0.9497 | 0.0054 | up

ERRNet/checkpoints/errnet_mixdata/errnet_050_00096650.pt
divov3+data transform                    
ceilnet_table2: LMSE: 0.0125 | NCC: 0.8818 | PSNR: 18.6178 | SSIM: 0.8318 |                  
real20: LMSE: 0.0173 | NCC: 0.9111 | PSNR: 24.3347 | SSIM: 0.8425 |                   
postcard: LMSE: 0.0031 | NCC: 0.9653 | PSNR: 24.7796 | SSIM: 0.9103 |                   
objects: LMSE: 0.0029 | NCC: 0.9844 | PSNR: 25.7775 | SSIM: 0.9119 |              
wild: LMSE: 0.0055 | NCC: 0.9500 | PSNR: 25.7647 | SSIM: 0.9069 | 


ERRNet/checkpoints/errnet_modern/errnet_best_psnr_val.pt
ceilnet_table2: LMSE: 0.0123 | NCC: 0.8850 | PSNR: 18.2306 | SSIM: 0.8300 |
real20: LMSE: 0.0170 | NCC: 0.8992 | PSNR: 24.3593 | SSIM: 0.8391 |
postcard: LMSE: 0.0032 | NCC: 0.9697 | PSNR: 25.2055 | SSIM: 0.9099 |
objects: LMSE: 0.0027 | NCC: 0.9861 | PSNR: 26.4564 | SSIM: 0.9104 |
wild: LMSE: 0.0056 | NCC: 0.9489 | PSNR: 25.2976 | SSIM: 0.9063 |
25.3297


CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 test_errnet.py \
--dataset all \
--name errnet_dinov3_scaled \
-r --icnn_path checkpoints/errnet_dinov3_scaled/errnet_latest.pt \
--hyper --feature_model_path /oldhome/zengyuqi/model/dinov3