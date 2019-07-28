# trajCamPose

In this project, we proposed a method to estimated the 3D camera pose (of a static surveillance camera) from 2D pedestrian trajectories.  We have access to a rough estimation of the real camera pose, generate synthetic pedestrian trajectories for training our regressor, apply the trained regressor with pedestrian extracted from real surveillance video.

![Visualization of the real ground plane and the re-projected ground plane.  The blue dotted plane in each image represents the real position of the ground plane, while the pink dotted plane represents the ground plane projected with the camera pose predicted by our NN regressor.](https://github.com/yanx001/trajCamPose/blob/master/experiments/result_visualization/ground_reprojection.png)

<!-- The code was written by [Yan Xu](https://github.com/yanx001). -->

## Prerequisites
Linux or MacOS
Python 3.6+
CPU or NVIDIA GPU + CUDA CuDNN

## Tutorial

### Generate synthetic data
As a first step, we first generate synthetic training data, given an estimated initial camera pose.

```
python generate_data.py --data_root 'experiments/towncenter' --traj_num 1 -- traj_len 35
```

### Train

```
python train.py --data_root 'experiments' --beta 500 --checkpoints_dir 'experiments/checkpoints' --num_epoch 100
```

### Test

```
python test.py --data_root 'experiments/towncenter'
```
