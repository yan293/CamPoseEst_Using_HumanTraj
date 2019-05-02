# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import sys, os
import pdb
sys.path.append('..')

from video_utils import *
from camera_pars import *
from simulator.camera import Camera


def stats_of_traj(traj_2d, cam):
	''' the stats of 1 trajectory, max, min, mean, std of step speed '''
	traj_3d = cam.project_im2ground(traj_2d)
	coord_diff = traj_3d[1:] - traj_3d[:-1]
	step_speeds = np.linalg.norm(coord_diff, ord=2, axis=1)
	speed_mean, speed_std = np.mean(step_speeds), np.std(step_speeds)
	speed_max, speed_min = np.max(step_speeds), np.min(step_speeds)
	# print('max: {}, min: {}, mean: {}, std: {}'.format(speed_max, speed_min, speed_mean, speed_std))
	return speed_mean, speed_std


def stats_across_trajs(trajs_2d, cam):
	''' the stats of many trajs of 1 camera, max, min, mean, std of pedestrian speeds '''
	speed_means, speed_stds = [], []

	for i in range(len(trajs_2d)):
		traj_2d = trajs_2d[i]
		if len(traj_2d) > 10:
			speed_mean, speed_std = stats_of_traj(traj_2d, cam)
			if speed_mean < 100:
				speed_means.append(speed_mean)
				speed_stds.append(speed_std)
				# print('mean: {}, std: {}'.format(speed_mean, speed_std))
	# print('mean speed across pedestrians: {}'.format(np.mean(speed_means)))
	return speed_means, speed_stds



def main():

	# cameras
	frame_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	camera_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	# camera_ids = [5]
	extrins = [CAM1_EXTRIN, CAM2_EXTRIN, CAM3_EXTRIN, CAM4_EXTRIN,
	           CAM5_EXTRIN, CAM6_EXTRIN, CAM7_EXTRIN, CAM8_EXTRIN, TOWNC_EXTRIN]
	intrins = [CAM1_INTRIN, CAM2_INTRIN, CAM3_INTRIN, CAM4_INTRIN,
	           CAM5_INTRIN, CAM6_INTRIN, CAM7_INTRIN, CAM8_INTRIN, TOWNC_INTRIN]
	distorts = [CAM1_DISTORT, CAM2_DISTORT, CAM3_DISTORT, CAM4_DISTORT, CAM5_DISTORT,
	            CAM6_DISTORT, CAM7_DISTORT, CAM8_DISTORT, TOWNC_DISTORT]

	# each camera
	data_save = {}
	for i in range(len(camera_ids)):

		# trajs
		cam_id, frame_id = camera_ids[i], frame_ids[i]
		if cam_id is 9:
			img = plt.imread('data/townCenter/frames/frame3902.jpg')
			traj_file = 'data/townCenter/testdata/data_test.pkl'
		else:
			img = plt.imread('data/dukeMTMC/frames/camera' + str(cam_id)
		                 + '_009_frames/frame' + str(frame_id) + '.jpg')
			traj_file = 'data/dukeMTMC/testdata/data_test' + str(cam_id) + '.pkl'
		traj_data = pkl.load(open(traj_file, 'rb'))
		trajs = traj_data['data']

		# camera
		cam = Camera(extrins[cam_id-1], in_pars=intrins[cam_id-1], distort_coeff=distorts[cam_id-1])
		speed_means, speed_stds = stats_across_trajs(trajs, cam)

		# save for visualization
		data_save[cam_id] = [speed_means, speed_stds]

		# # visualizer
		# visualizer = frameVisualizer(img, trajs=trajs)
		# visualizer.visualize_traj_in_frame()
		# plt.show()

	pkl.dump(data_save, open('../results/traj_stats.pkl', 'wb'))


if __name__ == '__main__':

	main()

