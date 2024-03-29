# trajCamPose

In this project, we proposed a method to estimated the 3D camera pose (of a static surveillance camera) from 2D pedestrian trajectories.  We have access to a rough estimation of the real camera pose, generate synthetic pedestrian trajectories for training our regressor, apply the trained regressor with pedestrian extracted from real surveillance video.

A visualization of the real ground plane and the re-projected ground plane is given in the following figure. The blue dotted plane in each image represents the real position of the ground plane, while the pink dotted plane represents the ground plane projected with the camera pose **_P_** predicted by our NN regressor.

<p align="center">
    <img src="./experiments/result_visualization/ground_reprojection.png" alt="ground reprojection"  width="750">
</p>

<!-- <p align="center">
    <img src="./experiments/result_visualization/ground_reprojection.png" alt="Sample"  width="500">
    <p align="center">
        <em>A visualization of the real ground plane and the re-projected ground plane. The blue dotted plane in each image represents the real position of the ground plane, while the pink dotted plane represents the ground plane projected with the camera pose $\boldsymbol{\tilde{\mathcal{P}}}$ predicted by our NN regressor.</em>
    </p>
</p> -->

<!-- The code was written by [Yan Xu](https://github.com/yanx001). -->

## Prerequisites
Linux or MacOS
Python 3.6+
CPU or NVIDIA GPU + CUDA CuDNN

## Tutorial

### Generate synthetic data
As a first step, we first generate synthetic training data, given an estimated initial camera pose.  The camera name is specified in the "data_root".

```
python generate_data.py --data_root 'experiments/towncenter' --traj_num 10 -- traj_len 31
```

### Train

The training command is given as follows:

```
python train.py  --exp_dir 'experiments/towncenter' --checkpoints_dir 'experiments/towncenter/checkpoints' --num_epoch 50 --beta 1
```

### Test

The test command is given as follows:

```
python test.py --exp_dir 'experiments/towncenter' --checkpoints_dir 'experiments/towncenter/checkpoints' --num_epoch 50
```
