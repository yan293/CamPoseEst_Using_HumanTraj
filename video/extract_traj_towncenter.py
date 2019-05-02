import os
import cv2
import sys
import pdb
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
sys.path.append('..')
from simulator.camera import Camera
from simulator.camera_parameters import *
from video_utils import frameVisualizer


def stats_of_traj(traj_2d, cam):
	''' the stats of 1 trajectory, max, min, mean, std of step speed '''
	traj_3d = cam.project_im2ground(traj_2d)
	coord_diff = traj_3d[1:] - traj_3d[:-1]
	step_speeds = np.linalg.norm(coord_diff, ord=2, axis=1)
	speed_mean, speed_std = np.mean(step_speeds), np.std(step_speeds)
	speed_max, speed_min = np.max(step_speeds), np.min(step_speeds)
	return speed_mean, speed_std


def extract_traj_video(traj_annot_file,  # bounding box path
                       cam_par,
                       traj_short=0,  # trajectories longer than
                       traj_long=1e10,  # trajectories shorter than
                       angle_unit='radian',  # scale of extrinsics
                       save_dir=None,  # data save directory
                       detection=False,
                       traj_detect_file=None):
	''' Process trajectories from video. '''
	print('\nConvert video trajectories to test data...\n')

	if detection:
		trajs_dect = pkl.load(open(traj_detect_file, 'rb'))
		person_trajs = []
		for key in trajs_dect.keys():
			traj_i = trajs_dect[key][0::cam_par['FPS']]
			if len(traj_i) > 6:
				# remove speed too fast or too slow
				cam = Camera(cam_par)
				speed_mean, speed_std = stats_of_traj(traj_i, cam)
				if speed_mean < 3.:
					print(speed_mean)
					person_trajs.append(traj_i)
		print(''.format(len(person_trajs)))
	else:
		# raw data
		file  = open(traj_annot_file, 'r')
		person_frame_bbox = []
		for line in file:
			row_i = list(map(float, line.rstrip().split(',')))
			row_i_pick = [ row_i[i] for i in [0, 1, 8, 9, 10, 11] ]
			person_frame_bbox.append(row_i_pick)
		person_frame_bbox = np.array(person_frame_bbox)
		ind_sort = np.lexsort((person_frame_bbox[:, 1], person_frame_bbox[:, 0]))
		person_frame_bbox = person_frame_bbox[ind_sort]
		person_id = person_frame_bbox[:, 0].astype(int)
		frame_id = person_frame_bbox[:, 1].astype(int)

		# extract trajs by person id
		person_trajs = []
		for i in np.unique(person_id):

			# traj of i-th person
			ind_p_i = np.where(person_id==i)
			xs_i = np.mean(person_frame_bbox[ind_p_i][:, [2, 4]], axis=1)
			ys_i = np.squeeze(person_frame_bbox[ind_p_i, 5])
			traj_i = np.column_stack((xs_i, ys_i))

			# interpolate if frame id not consecutive
			frames_p_i = frame_id[ind_p_i]
			if len(traj_i > 1):
				traj_i_interp = []
				traj_i_interp.append(traj_i[0])
				for k in range(1, len(frames_p_i)):
					frame_diff = frames_p_i[k] - frames_p_i[k-1]
					if frame_diff == 1:
						traj_i_interp.append(traj_i[k])
					else:
						xs_iterp = np.linspace(traj_i[k-1][0], traj_i[k][0], frame_diff + 1)
						ys_iterp = np.linspace(traj_i[k-1][1], traj_i[k][1], frame_diff + 1)
						points_interp = np.vstack([xs_iterp, ys_iterp]).T
						traj_i_interp.append(points_interp[1:])
				traj_i = np.vstack(traj_i_interp)
				traj_i = traj_i[0::cam_par['FPS']]

			# remove short trajs
			person_speed_mean = []
			if len(traj_i) > 6:
				# remove speed too fast or too slow
				cam = Camera(cam_par)
				speed_mean, speed_std = stats_of_traj(traj_i, cam)
				if speed_mean < 3.:
					print(speed_mean)
					person_speed_mean.append(speed_mean)
					person_trajs.append(traj_i)

	# restrict traj length
	trajs = []
	longest_traj = person_trajs[0]
	for traj in person_trajs:
		if len(traj) >= traj_short and len(traj) <= traj_long:
			trajs.append(traj)
		if len(longest_traj) < len(traj):
			longest_traj = traj
	person_trajs = trajs

	# label
	extrins = cam_par['translation'] + cam_par['rotation']
	label = np.array([extrins for _ in range(len(person_trajs))])

	# save
	if save_dir is not None:
		data_save = {'data': person_trajs, 'label': label,
		             'cam_intrin': cam_par['intrinsics'],
		             'image_size': cam_par['img_size']}
		pkl.dump(data_save, open(os.path.join(save_dir, 'data_test.pkl'), 'wb'))
	return person_trajs


def main():

	cam_par = CAMERA_TOWNCENTER
	traj_annot_file = 'data/townCenter/ground_truth.top'
	traj_detect_file = 'data/townCenter/person_traj_detection.pkl'
	save_dir = '../experiments/towncenter'
	trajs = extract_traj_video(traj_annot_file,
	                           cam_par,
	                           traj_short=11,
	                           traj_long=1e10,
	                           angle_unit='radian',
	                           save_dir=save_dir,
	                           detection=True,
	                           traj_detect_file=traj_detect_file)

	# # --- plot the trajectory as a test
	# img = plt.imread('data/townCenter/frames/frame4399.jpg')
	# visualizer = frameVisualizer(img, trajs=trajs)
	# visualizer.visualize_traj_in_frame()
	# visualizer.visualize_traj()
	# plt.show()


if __name__=='__main__':

	main()
