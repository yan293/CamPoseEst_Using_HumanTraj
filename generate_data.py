# -*- coding: utf-8 -*-

import os
import pdb
import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from transforms3d.euler import euler2quat, quat2euler

from simulator.camera import Camera
from simulator.simulator_options import *
from simulator.camera_parameters import *
from simulator.visualizer import Visualizer
from simulator.generator import TrajGenerator


def sample_around_cam_extrins(cam0, samp_range, samp_interval, axes = 'szxz'):

	# sample center (extrinsic)
	cam0_trans = cam0.trans
	cam0_rotat_quat = cam0.rotat_quat
	cam0_rotat_euler = np.rad2deg(quat2euler(cam0_rotat_quat, axes=axes))
	view_angle = max(cam0.compute_visual_angle()) # two angles (x-axis, y-axis)

	# sample extrinsics
	tz_range = samp_range[0]
	rx_range = samp_range[1]
	ry_range = samp_range[2]
	rz_range = samp_range[3]
	tx = cam0_trans[0]
	ty = cam0_trans[1]
	tz = np.arange(tz_range[0], tz_range[1], samp_interval[-4]) + cam0_trans[2]
	rx = np.arange(rx_range[0], rx_range[1], samp_interval[-3]) + cam0_rotat_euler[0]
	ry = np.arange(ry_range[0], ry_range[1], samp_interval[-2]) + cam0_rotat_euler[1] # pitch
	rz = np.arange(rz_range[0], rz_range[1], samp_interval[-1]) + cam0_rotat_euler[2]

	# pick out extrinsics with reasonable pitch angle
	idx_pitch = ry >= min(90.+view_angle/2., cam0_rotat_euler[1])
	ry = ry[idx_pitch]
	sampled_extrin_sets = np.array(np.meshgrid(tx, ty, tz, rx, ry, rz)).T.reshape(-1, 6)

	# Euler to quaternion
	sampled_trans = sampled_extrin_sets[:, :3]
	sampled_rotat = sampled_extrin_sets[:, 3:]
	sampled_rotats_quat = []
	for i in range(len(sampled_rotat)):
		rotat_euler = np.deg2rad(sampled_rotat[i])
		rotat_quat = euler2quat(rotat_euler[0], rotat_euler[1], rotat_euler[2], axes=axes)
		sampled_rotats_quat.append(rotat_quat)
	sampled_rotats_quat = np.array(sampled_rotats_quat)
	sampled_extrin_sets = np.concatenate((sampled_trans, sampled_rotats_quat), axis=1)
	return sampled_extrin_sets


def generate_data(cam0, samp_range, samp_interval, traj_len=30, traj_num=10, speed=1.2, save_file=None):
	''' generate synthetic traj data and extrinsics label given estimated camera '''

	# sample extrinsics around ground truth
	extrins_sets = sample_around_cam_extrins(cam0, samp_range, samp_interval)

	# generate data and label
	data, label = [], []
	for idx, cam_extrin in enumerate(extrins_sets):

		print('Synthetic data for camera {}/{}.'.format(idx+1, len(extrins_sets)), end='\r')

		# camera parameters
		cam_trans = cam_extrin[:3]
		cam_rotat_quat = cam_extrin[3:]
		cam_pars = {
			'intrinsics': cam0.intrins,
			'distortion': cam0.distorts,
			'translation': cam_trans,
			'rotation': cam_rotat_quat,
			'img_size': cam0.img_size
		}

		# generate trajectories for the i-th camera
		cam_i = Camera(cam_pars)
		gen_i = TrajGenerator(cam_i)
		dict_i = gen_i.generate_trajs(traj_len=traj_len,
		                              traj_num=traj_num,
		                              human_speed=speed)
		data += dict_i['data']
		label += dict_i['label']
	label = np.array(label)

	# save data
	if save_file is not None:
		data_save = {'data': data, 'label': label,
		             'intrin_mat': cam0.in_mat,
		             'cam_extrin': cam0.extrins,
		             'image_size': cam0.img_size}
		pkl.dump(data_save, open(save_file, 'wb'))
	return data_save


def main():

	cams = {'towncenter': CAMERA_TOWNCENTER,
	        'duke1': CAMERA_DUKE_1,
	        'duke2': CAMERA_DUKE_2,
	        'duke3': CAMERA_DUKE_3,
	        'duke4': CAMERA_DUKE_4,
	        'duke5': CAMERA_DUKE_5,
	        'duke6': CAMERA_DUKE_6,
	        'duke7': CAMERA_DUKE_7,
	        'duke8': CAMERA_DUKE_8
	        }
	sim_opts = SimulationOptions().parse()
	cam_name = sim_opts.data_root.split('/')[-1]
	camera = Camera(cams[cam_name])
	print('camera: {}'.format(cam_name))
	samp_range = [sim_opts.tz_range, sim_opts.rx_range,
	              sim_opts.ry_range, sim_opts.rz_range]
	samp_interval = [sim_opts.tz_inteval, sim_opts.rx_inteval,
	                 sim_opts.ry_inteval, sim_opts.rz_inteval]
	if not os.path.exists(sim_opts.data_root): os.mkdir(sim_opts.data_root)
	data_file = os.path.join(sim_opts.data_root, 'data_train.pkl')
	generate_data(camera,
	              samp_range,
	              samp_interval,
	              save_file=data_file,
	              traj_len=sim_opts.traj_len,
	              traj_num=sim_opts.traj_num,
	              speed=sim_opts.human_speed)  # set human speed


if __name__ == '__main__':

	main()
