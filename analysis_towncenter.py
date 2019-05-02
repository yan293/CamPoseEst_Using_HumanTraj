# -*- coding: utf-8 -*-
import pickle as pkl
import numpy as np
import torch
import copy
import pdb
import os

from models.trajnet import  trajNetModelBidirect
from simulator.parameters import *
from generate_data import *
from train import train, load_data
from test import test
from video.extract_trajvideo_towncenter import extract_traj_video
from video.camera_pars import *

INPUT_SIZE = 2
OUTPUT_SIZE = 4
FPS = 25
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EXP_DIR = 'experiments/real_video/towncenter'

data_train_path = os.path.join(EXP_DIR, 'data_train.pkl')
# data_test_path = os.path.join(EXP_DIR, 'data_test.pkl')
data_test_path = os.path.join(EXP_DIR, 'data_test_real.pkl')
bbox_test_path = 'video/data/townCenter/ground_truth.top'
# grid_steps = [0.2, 2., 2., 2.]
grid_steps = [0.2, 2., 2., 2.]
cam_extrinsics = TOWNC_EXTRIN
cam_intrinsics = TOWNC_INTRIN
cam_distort = TOWNC_DISTORT


def check_data_exist(real_video=False):
	''' Check if data exisits. '''
	traj_lower, traj_upper = 11, 1e10
	traj_num_train, traj_num_test = 10, 1
	if not os.path.exists(data_train_path):
		generate_data(cam_extrinsics, cam_intrinsics, grid_steps, traj_len=15,
		              traj_num=traj_num_train, speed=PEDESTRIAN_SPEED,
		              save_root=EXP_DIR, data_use='train')
	# --- test data from real video
	if real_video:
		trajs = extract_traj_video(bbox_test_path, FPS, cam_extrinsics, cam_intrinsics, distort=cam_distort,
		                   label_dim=OUTPUT_SIZE, traj_lower=traj_lower,
		                   traj_upper=traj_upper, angle_unit='radian', save_dir=EXP_DIR)

		# debug
		for i in range(len(trajs)):
			traj = trajs[i]
			traj_1 = traj[:-1]
			traj_2 = traj[1:]
			diff = traj_2 - traj_1
			dist = np.linalg.norm(diff, axis=1)
			# print(dist)
			# pdb.set_trace()
	return None


def main():

	# load
	# check_data_exist(real_video=True)
	dataloader_train = load_data(data_train_path)
	dataloader_test = load_data(data_test_path)
	# print('Loading data...\n', pkl.load(open(data_test_path, 'rb'))['label'])

	# train
	if not os.path.exists(os.path.join(EXP_DIR, 'trained_model.pt')):
		model = trajNetModelBidirect(INPUT_SIZE, OUTPUT_SIZE).to(DEVICE)
		model = train(model, dataloader_train, save_path=EXP_DIR, record_log=True)
	else:
		model = torch.load(os.path.join(EXP_DIR, 'trained_model.pt')).eval()

	# test
	pred_err, pred_std = test(model, dataloader_test, angle_unit='radian', device=DEVICE)


if __name__ == '__main__':

	main()
