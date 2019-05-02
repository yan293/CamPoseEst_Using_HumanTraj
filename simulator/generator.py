# -*- coding: utf-8 -*-

import pdb
import numpy as np
import pickle as pkl
from matplotlib import path
from .simulator_options import *
from shapely.geometry import Polygon, Point
from .camera import Camera, IntrinsicCamera


class TrajGenerator():

	def __init__(self, camera):
		self.camera = camera

	def generate_1_traj(self, fov_vertices, v, v_pha_noise=10, traj_len=10, traj_select=True):
		'''
		given "fov_vertices", pedestrian speed "v", noise on speed v_pha "v_pha_noise"
		(degree), trajectory length "traj_len", generate a trajectory lying inside polygon
		'''
		if traj_len < 3:
			raise RuntimeError('Trajectory length must be greater than 3.')

		traj_in_fov = False
		v_pha_noise = np.deg2rad(v_pha_noise)
		while not traj_in_fov:

			# step 1: randomly sample a point inside polygon
			polygon = Polygon(fov_vertices)
			(x_min, y_min, x_max, y_max) = polygon.bounds
			while True:
				point = Point(np.random.uniform(x_min, x_max),
				              np.random.uniform(y_min, y_max))
				if polygon.contains(point): break
			mid_traj_pt = np.array([point.x, point.y])

			# step 2: generate 1 trajectory from the point
			traj_forw = []
			traj_backw = [mid_traj_pt]
			v_mag, v_pha = v[0], v[1]
			curr_pos_1, curr_pos_2 = mid_traj_pt, mid_traj_pt
			v_pha_1, v_pha_2 = v_pha, v_pha

			for i in range((traj_len - 1) // 2):

				# move forward from mid point
				v_pha_1 = v_pha_1 + v_pha_noise*2*(np.random.rand()-0.5)
				step_1 = np.array([v_mag*np.cos(v_pha_1), v_mag*np.sin(v_pha_1)])
				curr_pos_1 = curr_pos_1 - step_1
				traj_forw.append(curr_pos_1)

				# move backward from mid point
				v_pha_2 = v_pha_2 + v_pha_noise*2*(np.random.rand()-0.5)
				step_2 = np.array([v_mag*np.cos(v_pha_2), v_mag*np.sin(v_pha_2)])
				curr_pos_2 = curr_pos_2 + step_2
				traj_backw.append(curr_pos_2)

			if (traj_len-1)%2:  # make sure "traj_num" is correct
				v_pha_2 = v_pha_2 + v_pha_noise*2*(np.random.rand()-0.5)
				step_2 = np.array([v_mag*np.cos(v_pha_2), v_mag*np.sin(v_pha_2)])
				curr_pos_2 = curr_pos_2 + step_2
				traj_backw.append(curr_pos_2)

			traj_forw, traj_backw = np.array(traj_forw), np.array(traj_backw)
			traj_forw = np.flip(traj_forw, axis=0)
			traj = np.concatenate((traj_forw, traj_backw))

			# step 3: remove partition outside polygon (does this really impact training?)
			if traj_select:
				step_idx = np.arange(len(traj))
				path0 = path.Path(fov_vertices)
				inside_poly = path0.contains_points(traj)

				step_in_poly = step_idx[inside_poly]
				step_diff = step_in_poly[1:] - step_in_poly[:-1]
				if all(step_diff <= 1) and len(traj[inside_poly]) >= 7:
					traj_in_fov = True
					return traj[inside_poly]
				# if len(traj[inside_poly]) == traj_len:
				# 	return traj[inside_poly]

	def generate_trajs(self, traj_len=50, traj_num=100, human_speed=1.2):
		''' generate "traj_num" of 2D image trajectories for the camera '''

		cam = self.camera
		trajs, extrins = [], []
		fov_vertices, _ = cam.compute_ground_fov()
		cam_extrin_pars = np.array(list(cam.trans) + list(cam.rotat_quat))

		# generate trajs
		for i in range(traj_num):
			v = [human_speed, 2 * np.pi * np.random.rand()]
			traj_ground = self.generate_1_traj(fov_vertices, v, v_pha_noise=19, traj_len=traj_len)
			traj_image = cam.project_ground2im(traj_ground)
			trajs.append(traj_image)
			extrins.append(cam_extrin_pars)
		data_dict = {'data': trajs, 'label': extrins}
		return data_dict

	# def generate_trajs(self, traj_len=50, traj_num=100, angle='radian'):
	# 	''' generate "traj_num" of 2D image trajectories for the camera '''

	# 	cam = self.camera
	# 	trajs, extrins = [], []
	# 	fov_vertices, _ = cam.compute_ground_fov()

	# 	# scale of rotation angle
	# 	if angle=='radian':
	# 		cam_extrin_pars = np.array([cam.tx, cam.ty, cam.tz,
	# 		                            cam.rx, cam.ry, cam.rz])
	# 	elif angle=='degree':
	# 		cam_extrin_pars = np.array([cam.tx, cam.ty, cam.tz] +
	# 		           list(np.rad2deg([cam.rx, cam.ry, cam.rz])))
	# 	# generate trajs
	# 	for i in range(traj_num):
	# 		v = [self.human_speed, 2 * np.pi * np.random.rand()]
	# 		traj_ground = self.generate_1_traj(fov_vertices, v, v_pha_noise=19, traj_len=traj_len)
	# 		traj_image = cam.project_ground2im(traj_ground)
	# 		trajs.append(traj_image)
	# 		extrins.append(cam_extrin_pars[-4:])
	# 	data_dict = {'data': trajs, 'label': extrins, 'angle_scale': angle}
	# 	return data_dict
