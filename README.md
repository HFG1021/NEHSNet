# NEHSNet: A Node-Edge-Hyperedge Synergy Network for 2D-to-3D Human Pose Estimation

| ![skating](figure/skating.gif)  | ![anime](figure/dancing.gif) |
| ------------- | ------------- |

This is the official implementation of the approach described in the paper:

> [**NEHSNet: A Node-Edge-Hyperedge Synergy Network for 2D-to-3D Human Pose Estimation**]           
> Fengguang Hu, Tuqian Zhang, Yongkang Hu, Xinrui Yu, Shumei Bao,        


<p align="center"><img src="figure/pipeline.png" width="80%" alt="" /></p>

## Installation

NEHSNet is tested on Ubuntu 18 with Pytorch 1.7.1 and Python 3.9. 
- Create a conda environment: ```conda create -n graphmlp python=3.9```
- Install PyTorch 1.7.1 and Torchvision 0.8.2 following the [official instructions](https://pytorch.org/)
- ```pip3 install -r requirements.txt```
  
## Dataset setup

Please download the dataset from [Human3.6M](http://vision.imar.ro/human3.6m/) website and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) website, and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./dataset' directory). 
Or you can download the processed data from [here](https://drive.google.com/drive/folders/18mvXIZ98LKGAqDFpRsNVvCRonVBAlgoX?usp=share_link). 

```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_3d_3dhp.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
|   |-- data_2d_3dhp.npz
```

## Download pretrained model

The pretrained model can be found in [here](https://pan.baidu.com/s/1aL_2JR7snhvfNwOSEFS0dg?pwd=dnvv), please download it and put it in the './checkpoint/pretrained' directory. 

## Test the model

To test a 1-frame NEHSNet model:

```bash
# Human3.6M
python main.py --test --previous_dir 'checkpoint/pretrained/1' --frames 1

# MPI-INF-3DHP
python main.py --test --previous_dir 'checkpoint/pretrained/1' --frames 1 --dataset '3dhp'
```

To test a 1-frame NEHSNet model with refine module on Human3.6M:
```bash
python main.py --test --previous_dir 'checkpoint/pretrained/1/refine' --frames 1 --refine --refine_reload
```


## Train the model

To train a 1-frame GraphMLP model on Human3.6M:

```bash
# Train from scratch
python main.py --frames 1 --batch_size 256

# After training for 20 epochs, add refine module
python main.py --frames 1 --batch_size 256 --refine --lr 1e-5 --previous_dir [your best model saved path]
```

## Demo
First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in the './demo/lib/checkpoint' directory. 
Then, you need to put your in-the-wild videos in the './demo/video' directory. 

Run the command below:
```bash
# Run the command below:
python demo/vis.py --video sample_video.mp4

# Or run the command with the fixed z-axis:
python demo/vis.py --video sample_video.mp4 --fix_z
```

Sample demo output:

| ![regression](figure/regression.gif) |
| ------------- |

<!-- <p align="center"><img src="figure/sample_video.gif" width="60%" alt="" /></p>

Or run the command below:
```bash
python demo/vis.py --video sample_video.mp4 --fix_z
```

Sample demo output:

<p align="center"><img src="figure/sample_video_fix_z.gif" width="60%" alt="" /></p> -->


## Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes. 

- [ST-GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D)
- [GraphMLP](https://github.com/Vegetebird/GraphMLP)

