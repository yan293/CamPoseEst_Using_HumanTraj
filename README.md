# trajCamPose

In this project, we proposed a method to estimated the 3D camera pose (of a static surveillance camera) from 2D pedestrian trajectories.  We have access to a rough estimation of the real camera pose, generate synthetic pedestrian trajectories for training our system, apply the trained system with pedestrian extracted from real surveillance video.

The code was written by [Yan Xu](https://github.com/yanx001).

## Prerequisites
Linux or MacOS  
Python 3.6+  
CPU or NVIDIA GPU + CUDA CuDNN

## Tutorial

### Generate synthetic data

```
python generate_data.py --data_root 'experiments/towncenter' --traj_num 1 -- traj_len 35
```

### Train

```python
python train.py --data_root 'experiments' --beta 500 --checkpoints_dir 'experiments/checkpoints' --num_epoch 100
```

### Test

```python
python test.py --data_root 'experiments/towncenter'
```
