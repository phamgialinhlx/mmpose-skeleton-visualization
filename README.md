# **Prerequisites**

**Step 0. Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).**

**Step 1. Create a conda environment and activate it.**

```bash
conda create --name openmmlab python**=**3.8 -y
conda activate openmmlab
```

**Step 2. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.**

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

# Installation

**Step 0. Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).**

```bash
pip install -U openmim
mim install mmcv-full
```

**Step 1. Install MMPose.**

```bash
pip install mmpose
```

# Verify the installation

> To verify that MMPose is installed correctly, you can run an inference demo with the following steps.
> 

**Step 1. We need to download config and checkpoint files.**

```bash
mim download mmpose --config associative_embedding_hrnet_w32_coco_512x512  --dest .
```

**Step 2. Verify the inference demo.**

```bash
from mmpose.apis import (init_pose_model, inference_bottom_up_pose_model, vis_pose_result)

config_file = 'associative_embedding_hrnet_w32_coco_512x512.py'
checkpoint_file = 'hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
pose_model = init_pose_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

image_name = 'demo/persons.jpg'
# test a single image
pose_results, _ = inference_bottom_up_pose_model(pose_model, image_name)

# show the results
vis_pose_result(pose_model, image_name, pose_results, out_file='demo/vis_persons.jpg')
```

## Download mmdet and pretrained

Download mmdet:

```bash
pip install mmdet
```

Pretrain:

- Pretrain cho  mmdet
    
    ```bash
    wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    ```
    
- Pretrain cho mmpose
    
    ```bash
    wget https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth
    wget https://download.openmmlab.com/mmpose/top_down/resnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
    ```
    

Other package:

```bash
pip install einops
```

# Inference

Example:

```bash
python skeleton_pose.py mmdet_cfg.py /data/pill/emotion/mmpose/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth mmpose_cfg.py /data/pill/emotion/mmpose/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth  --video-path /data/baby/Workspace/tungch/dnp/00_poc/data/output.mp4 --out-video-root vis_results/o --out-video-bg
```

`mmdet_cfg.py`:

- filepath to the pretrained
- ***meaning:*** file config for mmdetection
- ***default:*** faster_rcnn_r50_fpn_coco

`mmpose_cfg.py`:

- filepath to the pretrained
- ***meaning*:** file config for mmpose
- ***default:*** hrnet_w48_coco_wholebody_384x288_dark

`video-path`: 

- filepath to the video
- **type:** str

`out-video-root`:

- target path

`out-video-bg`:

- action=’store_true’
- ***default:*** False

`bg-img`:

- Must enable the out-video-bg option to use this feature
- ***default:*** white color backgroundSource code

---
