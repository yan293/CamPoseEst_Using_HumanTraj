import os
import cv2
import pdb
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


def rotationMatToEuler():

	return None


def video2frame(video_path, frame_save_dir):
	'''
	This function load in a video and split it into frames and store the frames to the path
	Input: Path to the video
	Output: None
	'''
	vidcap = cv2.VideoCapture(video_path)
	success, frame = vidcap.read()
	count = 0
	if not os.path.exists(frame_save_dir):
		os.makedirs(frame_save_dir)
	while success:
		cv2.imwrite(os.path.join(frame_save_dir + 'frame{}.jpg'.format(count)), frame)
		success, frame = vidcap.read()
		print('Read a new frame{}: '.format(count), success)
		count += 1
	return None


class frameVisualizer():
	def __init__(self, img, trajs=None):
		self.img = img
		self.trajs = trajs

	def visualize_frame(self):
		fig = plt.figure()
		ax = fig.add_subplot(111, aspect='auto')
		ax.imshow(self.img)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		fig.tight_layout()
		fig.savefig('result/only_frame.png', bbox_inches='tight', pad_inches=0)
		return None

	def visualize_traj(self):
		fig = plt.figure()
		ax = fig.add_subplot(111, aspect='auto')
		color = 'tab:blue'
		if self.trajs is not None:
			for i in range(len(self.trajs)):
				traj = self.trajs[i]
				x, y = traj[:, 0], traj[:, 1]
				ax.plot(x, y, '-', markersize=1, c=color, alpha=0.5)
		ax.set_xlim(0, self.img.shape[1])
		ax.set_ylim(self.img.shape[0], 0)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.axis('off')
		fig.tight_layout()
		fig.savefig('result/only_traj.png', bbox_inches='tight', pad_inches=0)
		return None

	def visualize_traj_in_frame(self):

		# visualize frame
		fig = plt.figure()
		ax = fig.add_subplot(111, aspect='auto')
		ax.imshow(self.img, alpha=1.)

		# visualize trajectories
		color = 'tab:blue'
		if self.trajs is not None:
			for i in range(len(self.trajs)):
				traj = self.trajs[i]
				x, y = traj[:, 0], traj[:, 1]
				ax.plot(x, y, '-', markersize=1, c=None, alpha=1.)
		ax.set_xlim(0, self.img.shape[1])
		ax.set_ylim(self.img.shape[0], 0)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.axis('off')
		fig.tight_layout()
		fig.savefig('result/traj_in_frame.png', bbox_inches='tight', pad_inches=0)
		return None
