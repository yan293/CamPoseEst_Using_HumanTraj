# -*- coding: utf-8 -*-

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pdb


def plot_traj(ax, traj, color='tab:blue'):
	'''
	Plot trajectory
	Input:  traj(trajectory, Nx2 array), ax(the fig)
	Output: no output
	'''
	Xs = traj[:, 0]
	Ys = traj[:, 1]
	ax.plot(Xs, Ys, '-', markersize=2, c=color)


class Visualizer():

	def __init__(self, camera, trajs_2d=None):
		self.camera = camera
		self.trajs_2d = trajs_2d


	def visualize_image(self, save_name=None):
		'''
		visualize image plane, and 2d trajectories if "2d_trajs" is not "None"
		'''
		img_size = self.camera.img_size

		# -- image border and axises
		fig = plt.figure()
		ax = fig.add_subplot(111, aspect='equal')
		ax.set_xlim(np.array([0., img_size[0]]))
		ax.set_ylim(np.array([0., img_size[1]]))
		ax.add_patch(patches.Rectangle((0., 0.),  img_size[0], img_size[1],
		                               lw=1., ls=None, fill=False))
		ax.arrow(img_size[0] / 2., img_size[1] / 2., 150, 0., head_width=30.,
		         head_length=30., fc='k', ec='k', zorder=10)
		ax.arrow(img_size[0] / 2., img_size[1] / 2., 0., 150., head_width=30.,
		         head_length=30., fc='k', ec='k', zorder=10)

		# -- trajectories
		if self.trajs_2d is not None:
			for traj in self.trajs_2d:
				plot_traj(ax, traj, color=None)

		# ax.get_xaxis().set_visible(False)
		# ax.get_yaxis().set_visible(False)
		# ax.set_axis_off()
		# ax.axis('off')
		# plt.xticks([])
		# plt.yticks([])
		# plt.xlabel(r"$\bf{x}$" + " (pixel)")
		# plt.ylabel(r"$\bf{y}$" + " (pixel)")
		fig.tight_layout()
		if save_name is None:
			fig.savefig('results/observed_image.png', bbox_inches='tight')
		else:
			fig.savefig(save_name, bbox_inches='tight')
		return None


	def visualize_fov(self, save_name=None):
		'''
		visualize ground fov, and 3d trajectories if "3d_trajs" is not "None"
		'''
		# -- ground fov
		fov_corners, fov_center = self.camera.compute_ground_fov()
		x_min, x_max = np.min(fov_corners[:, 0]), np.max(fov_corners[:, 0])
		y_min, y_max = np.min(fov_corners[:, 1]), np.max(fov_corners[:, 1])

		# -- border
		fig = plt.figure()
		ax = fig.add_subplot(111, aspect='equal')
		ax.set_xlim(np.array([x_min, x_max]))
		ax.set_ylim(np.array([y_min, y_max]))
		ax.add_patch(patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
		                               lw=3, ls=None, fill=True, color='tab:gray'))
		ax.add_patch(patches.Polygon(xy=list(zip(fov_corners[:,0], fov_corners[:,1])),
		                             ls='-', lw=1.5, color='w', alpha=1., fill=True))

		# -- axises
		x_unit = fov_center[1, :] - fov_center[0, :]
		x_unit = x_unit / np.linalg.norm(x_unit, ord=2)
		y_unit = fov_center[2, :] - fov_center[0, :]
		y_unit = y_unit / np.linalg.norm(y_unit, ord=2)
		x_unit = x_unit * min(x_max-x_min, y_max-y_min) * 0.15
		y_unit = y_unit * min(x_max-x_min, y_max-y_min) * 0.15
		ax.arrow(fov_center[0, 0],fov_center[0, 1], x_unit[0], x_unit[1],
				 head_width=.5, head_length=.5, fc='k', ec='k', zorder=10)
		ax.arrow(fov_center[0, 0], fov_center[0, 1], y_unit[0], y_unit[1],
				 head_width=.5, head_length=.5, fc='k', ec='k', zorder=10)

		# -- trajectories
		if self.trajs_2d is not None:
			trajs_3d = []
			for traj_2d in self.trajs_2d:
				traj_3d = self.camera.project_im2ground(traj_2d)
				plot_traj(ax, traj_3d, color=None)
		plt.xlabel(r"$\bf{X}$" + " (meter)")
		plt.ylabel(r"$\bf{Y}$" + " (meter)")
		if save_name is None:
			fig.savefig('results/fov_top_view.png', bbox_inches='tight')
		else:
			fig.savefig(save_name, bbox_inches='tight')
		return None
