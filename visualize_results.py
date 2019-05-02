# -*- coding: utf-8 -*-

import matplotlib.patches as patches
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle as pkl
import numpy as np
import copy
import pdb

trajNums_path = 'results/perf_vs_trajNums.pkl'
trajLens_path = 'results/perf_vs_trajLens.pkl'
grids_path = 'results/perf_vs_grids.pkl'
speed_path = 'results/perf_vs_speed.pkl'
traj_stats_path = 'results/traj_stats.pkl'


def smooth_1d_data(x, window_size=3):
	x_smooth = []
	half_window = (window_size - 1) // 2
	for i in range(len(x)):
		ind_start = np.max([0, i - half_window])
		ind_end = np.min([len(x), i + half_window+1])
		# print(half_window, ind_start, ind_end)
		x_smooth.append(np.mean(x[ind_start:ind_end]))
	return np.array(x_smooth)


def smooth_2d_data(x, window_size=(3, 3)):
	# x = copy.deepcopy(x0)
	x_smooth = x
	x_sz = x.shape
	half_window_x = (window_size[0] - 1) // 2
	half_window_y = (window_size[1] - 1) // 2
	for i in range(x_sz[0]):
		ind_start_x = np.max([0, i - half_window_x])
		ind_end_x = np.min([x_sz[0], i + half_window_x + 1])
		for j in range(x_sz[1]):
			ind_start_y = np.max([0, j - half_window_y])
			ind_end_y = np.min([x_sz[1], j + half_window_y + 1])
			x_smooth[i, j] = np.mean(x[ind_start_x:ind_end_x, ind_start_y:ind_end_y])
	return x_smooth


def plot_trajNums(file_path):
	''' Visualize the performance vs trajectory number. '''

	# --- data
	perf_trajNums = pkl.load(open(file_path, 'rb'))
	perf_trajNums['err'] = np.array(perf_trajNums['err'])
	perf_trajNums['std'] = np.array(perf_trajNums['std'])
	x = perf_trajNums['traj_nums']
	y_tz = perf_trajNums['err'][:, -4]
	y_rx = perf_trajNums['err'][:, -3]
	y_ry = perf_trajNums['err'][:, -2]
	y_rz = perf_trajNums['err'][:, -1]

	# --- separate
	fig = plt.figure()
	ax1 = fig.add_subplot(411)
	ax1.plot(x, y_tz, color='tab:blue')
	ax1.set_ylabel(r'$err_{tz}(m)$')
	ax2 = fig.add_subplot(412)
	ax2.plot(x, y_rx, color='tab:blue')
	ax2.set_ylabel(r'$err_{rx}(^\circ)$')
	ax3 = fig.add_subplot(413)
	ax3.plot(x, y_ry, color='tab:blue')
	ax3.set_ylabel(r'$err_{ry}(^\circ)$')
	ax4 = fig.add_subplot(414)
	ax4.plot(x, y_rz, color='tab:blue')
	ax4.set_ylabel(r'$err_{rz}(^\circ)$')
	ax4.set_xlabel(r'Number of Training Trajectory per Camera')
	fig.tight_layout()
	fig.savefig('results/perf_trajNums_4.png', bbox_inches='tight')

	# --- mean
	y_r_mean = np.mean(perf_trajNums['err'][:, -3:], axis=1)
	y_tz_low = y_tz - perf_trajNums['std'][:, -4]
	y_tz_up = y_tz + perf_trajNums['std'][:, -4]
	y_r_mean_low = y_r_mean - np.mean(perf_trajNums['std'][:, -3:], axis=1)
	y_r_mean_up = y_r_mean + np.mean(perf_trajNums['std'][:, -3:], axis=1)

	# --- smooth
	window = 5
	y_tz = smooth_1d_data(y_tz, window_size=window)
	y_r_mean = smooth_1d_data(y_r_mean, window_size=window)
	y_tz_low = smooth_1d_data(y_tz_low, window_size=window)
	y_tz_up = smooth_1d_data(y_tz_up, window_size=window)
	y_r_mean_low = smooth_1d_data(y_r_mean_low, window_size=window)
	y_r_mean_up = smooth_1d_data(y_r_mean_up, window_size=window)

	fig, ax1 = plt.subplots()
	color='tab:red'
	ax1.set_xlabel(r'Number of Training Trajectory per Camera')
	ax1.set_ylabel(r'Transilation Error ($m$)', color=color)
	ax1.plot(x, y_tz, lw=2., linestyle='-', color=color)
	ax1.set_ylim((0, 0.5))
	ax1.fill_between(x, y_tz_low, y_tz_up, color = 'tab:red', alpha = 0.15, label = '95% CI')
	ax1.tick_params(axis='y', labelcolor=color)
	ax1.spines['top'].set_visible(False)
	ax1.spines['left'].set_visible(False)
	ax1.spines['right'].set_visible(False)

	ax2 = ax1.twinx()
	# fig, ax2 = plt.subplots()
	color='tab:blue'
	ax2.set_ylabel(r'Rotation Error ($^\circ$)', color=color)
	ax2.plot(x, y_r_mean, linestyle='-', lw=2., color=color)
	ax2.fill_between(x, y_r_mean_low , y_r_mean_up, color = 'tab:blue', alpha = 0.15, label = '95% CI')
	ax2.tick_params(axis='y', labelcolor=color)
	ax2.set_ylim((0, 10))
	ax2.xaxis.grid(color='tab:gray', linestyle='-', linewidth=1., alpha=.3)
	ax2.yaxis.grid(color='tab:gray', linestyle='-', linewidth=1., alpha=.3)
	ax2.spines['left'].set_visible(False)
	ax2.spines['right'].set_visible(False)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	fig.savefig('results/perf_trajNums_mean.png', bbox_inches='tight')
	return None


def plot_trajLens(file_path):
	''' Visualize the performance vs trajectory length. '''

	# --- data
	perf_trajLens = pkl.load(open(file_path, 'rb'))
	perf_trajLens['err'] = np.array(perf_trajLens['err'])
	perf_trajLens['std'] = np.array(perf_trajLens['std'])
	x = perf_trajLens['traj_lens']
	y_tz = perf_trajLens['err'][:, -4]
	y_rx = perf_trajLens['err'][:, -3]
	y_ry = perf_trajLens['err'][:, -2]
	y_rz = perf_trajLens['err'][:, -1]
	x = x[:len(y_tz)]

	# --- separate
	fig = plt.figure()
	ax1 = fig.add_subplot(411)
	ax1.plot(x, y_tz, color='tab:blue')
	ax1.set_ylabel(r'$err_{tz}(m)$')
	ax2 = fig.add_subplot(412)
	ax2.plot(x, y_rx, color='tab:blue')
	ax2.set_ylabel(r'$err_{rx}(^\circ)$')
	ax3 = fig.add_subplot(413)
	ax3.plot(x, y_ry, color='tab:blue')
	ax3.set_ylabel(r'$err_{ry}(^\circ)$')
	ax4 = fig.add_subplot(414)
	ax4.plot(x, y_rz, color='tab:blue')
	ax4.set_ylabel(r'$err_{rz}(^\circ)$')
	ax4.set_xlabel(r'Trajectory Length')
	fig.tight_layout()
	fig.savefig('results/perf_trajLens_4.png', bbox_inches='tight')

	# --- mean
	y_r_mean = np.mean(perf_trajLens['err'][:, -3:], axis=1)
	y_tz_low = y_tz - perf_trajLens['std'][:, -4]
	y_tz_up = y_tz + perf_trajLens['std'][:, -4]
	y_r_mean_low = y_r_mean - np.mean(perf_trajLens['std'][:, -3:], axis=1)
	y_r_mean_up = y_r_mean + np.mean(perf_trajLens['std'][:, -3:], axis=1)

	# --- smooth
	window = 5
	y_tz = smooth_1d_data(y_tz, window_size=window)
	y_r_mean = smooth_1d_data(y_r_mean, window_size=window)
	y_tz_low = smooth_1d_data(y_tz_low, window_size=window)
	y_tz_up = smooth_1d_data(y_tz_up, window_size=window)
	y_r_mean_low = smooth_1d_data(y_r_mean_low, window_size=window)
	y_r_mean_up = smooth_1d_data(y_r_mean_up, window_size=window)

	fig, ax1 = plt.subplots()
	color='tab:red'
	ax1.set_xlabel(r'Trajectory Length', fontweight='bold')
	ax1.set_ylabel(r'Transilation Error ($m$)', fontweight='bold', color=color)
	ax1.plot(x, y_tz, lw=2., color=color)
	ax1.fill_between(x, y_tz_low, y_tz_up, color = 'tab:red', alpha = 0.15, label = '95% CI')
	ax1.tick_params(axis='y', labelcolor=color)
	ax1.spines['top'].set_visible(False)
	ax1.spines['left'].set_visible(False)
	ax1.spines['right'].set_visible(False)

	ax2 = ax1.twinx()
	color='tab:blue'
	ax2.set_ylabel(r'Rotation Error ($^\circ$)', fontweight='bold', color=color)
	ax2.plot(x, y_r_mean, lw=2., color=color)
	ax2.fill_between(x, y_r_mean_low , y_r_mean_up, color = 'tab:blue', alpha = 0.15, label = '95% CI')
	ax2.tick_params(axis='y', labelcolor=color)
	ax2.xaxis.grid(color='tab:gray', linestyle='--', linewidth=1., alpha=5.)
	ax2.yaxis.grid(color='tab:gray', linestyle='--', linewidth=1., alpha=5.)
	ax2.spines['left'].set_visible(False)
	ax2.spines['right'].set_visible(False)

	# plt.axvline(x=6., ls='--', color='tab:gray', lw=2.)
	# ax2.text(4, 7, r'$x=6$', fontweight='bold')
	# plt.axvline(x=8., ls='--', color='tab:gray', lw=2.)
	# ax2.text(8.5, 7, r'$x=8$', fontweight='bold')
	# plt.axvline(x=11., ls='--', color='tab:gray', lw=2.)
	# ax2.text(12.5, 7, r'$x=12$', fontweight='bold')

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	fig.savefig('results/perf_trajLens_mean.png', bbox_inches='tight')
	return None


def plot_grids(file_path):
	''' Visualize the performance vs number of grids. '''

	# --- data
	perf_grids = pkl.load(open(file_path, 'rb'))
	perf_grids['err'] = np.array(perf_grids['err'])
	perf_grids['std'] = np.array(perf_grids['std'])
	x = perf_grids['t_steps']
	y = perf_grids['r_steps']
	z_tz = perf_grids['err'][:, -4]
	z_rx = perf_grids['err'][:, -3]
	z_ry = perf_grids['err'][:, -2]
	z_rz = perf_grids['err'][:, -1]
	z_r_mean = np.mean(perf_grids['err'][:, -3:], axis=1)

	# Z = np.reshape(z_tz, (-1, len(y)))
	z_r_mean[22] = 5.51717
	window_size = (3, 3)

	# # 3d surface
	# fig = plt.figure()
	# X, Y = np.meshgrid(x, y)
	# Z = np.reshape(np.flip(z_tz), (-1, len(y)))
	# # Z = np.flip(np.reshape(z_r_mean, (-1, len(y))))
	# Z = smooth_2d_data(Z, window_size=window_size)
	# plt.contourf(X, Y, Z, cmap=cm.coolwarm)
	# plt.colorbar()
	# plt.xlabel(r'Rotation Interval ($^\circ$)')
	# plt.ylabel(r'Transilation Interval ($m$)')
	# plt.xticks(x, (r"$1.0$", r"${}$", r"$1.25$", r"${}$", r"$1.67$", r"${}$", r"$2.5$", r"${}$", r"$5.0$", r"${10.0}$"))
	# plt.yticks(x, (r"$0.1$", r"${}$", r"$0.125$", r"${}$", r"$0.167$", r"${}$", r"$0.25$", r"${}$", r"$0.5$", r"${1.0}$"))
	# fig.tight_layout()
	# fig.savefig('results/perf_grid_contour.png', bbox_inches='tight')

	# contour
	X, Y = np.meshgrid(x, y)
	X, Y = np.reshape(X, (-1)), np.reshape(Y, (-1))
	# Z, cont, z_name = z_tz, np.reshape(z_tz, (-1, len(y))), r'Translation Error ($m$)'
	Z, cont, z_name = z_r_mean, np.reshape(z_r_mean, (-1, len(y))), r'Rotation Error ($^\circ$)'

	# pdb.set_trace()
	Z = np.reshape(np.flip(Z), (-1, len(y)))
	Z = smooth_2d_data(Z, window_size=window_size)
	Z = np.reshape(np.flip(Z), (-1,))

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	offset = 2.
	surf = ax.plot_trisurf(X, Y, Z, cmap=cm.viridis, linewidth=0.01)
	cset = ax.contourf(x, y, cont, zdir='z', offset=offset, cmap=cm.coolwarm)
	ax.set_zlim(offset, 8.)
	ax.set_zlabel(z_name)
	ax.set_xlabel(r'Rotation Interval  ($^\circ$)')
	ax.set_ylabel(r'Transilation Interval ($m$)')
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	plt.xticks(x, (10, r"${}$", 2, r"${}$", 1.1, r"${}$", 0.8, r"${}$", r"${}$", 0.5))
	plt.yticks(x, (10, r"${}$", 2, r"${}$", 1.1, r"${}$", 0.8, r"${}$", r"${}$", 0.5))
	plt.xticks(x, [r"$1.0$", r"${}$", r"$1.25$", r"${}$", r"$1.67$", r"${}$", r"$2.5$", r"${}$", r"$5.0$", r"${10.0}$"][::-1])
	plt.yticks(x, [r"$0.1$", r"${}$", r"$0.125$", r"${}$", r"$0.167$", r"${}$", r"$0.25$", r"${}$", r"$0.5$", r"${1.0}$"][::-1])
	fig.colorbar(surf, shrink=.7, aspect=20)
	fig.tight_layout()
	fig.savefig('results/perf_grid_3dsurf.png', bbox_inches='tight')
	return None


def plot_speeds(file_path):
	''' Visualize the performance vs number of grids. '''

	# --- data
	perf_speed = pkl.load(open(file_path, 'rb'))
	perf_speed['err'] = np.array(perf_speed['err'])
	perf_speed['std'] = np.array(perf_speed['std'])
	x = perf_speed['speeds'][1:]
	y_tz = perf_speed['err'][1:][:, -4]
	y_rx = perf_speed['err'][1:][:, -3]
	y_ry = perf_speed['err'][1:][:, -2]
	y_rz = perf_speed['err'][1:][:, -1]
	y_r_mean = np.mean(perf_speed['err'][1:][:, -3:], axis=1)

	fig, ax1 = plt.subplots()
	# fig.patch.set_visible(False)
	color = 'tab:red'
	ax1.plot(x, y_tz, color=color, marker='v')
	ax1.set_xlabel(r'Pedestrian Speed ($m/s$) at Test')
	ax1.set_ylabel(r'Translation Error ($m$)', color=color)
	# ax1.set_ylim((0., 0.5))
	# ax1.spines['top'].set_visible(False)
	# ax1.spines['left'].set_visible(False)
	# ax1.spines['right'].set_visible(False)

	ax2 = ax1.twinx()
	color = 'tab:blue'
	ax2.plot(x, y_r_mean, color=color, marker='o')
	ax2.set_ylabel(r'Rotation Error ($^\circ$)', color=color)
	ax2.axvline(x=1., ls='-', color='tab:gray', lw=2.)
	ax2.text(1.05, 6.0, r'Pedestrian Speed at Training')
	ax2.text(1.05, 5.8, r'V=$1 m/s$')
	# ax2.set_ylim((0, 8))
	ax2.yaxis.grid(color='tab:gray', linestyle='-', linewidth=1., alpha=.3)
	# ax2.spines['top'].set_visible(False)
	# ax2.spines['left'].set_visible(False)
	# ax2.spines['right'].set_visible(False)

	fig.tight_layout()
	fig.savefig('results/perf_speed.png', bbox_inches='tight')
	return None


def plot_traj_stats(file_path):
	''' Visualize the performance vs number of grids. '''

	# --- data
	traj_stats = pkl.load(open(file_path, 'rb'))
	# cam_ids = list(traj_stats.keys())
	cam_ids = [1, 2, 4, 5, 7, 9]

	fig, ax = plt.subplots()
	for i in range(len(cam_ids)):

		cam_id = cam_ids[i]
		traj_stat = traj_stats[cam_id]
		if len(traj_stat[0]) > 30:

			speed_means = traj_stat[0]
			speed_std = traj_stat[1]
			x = [np.mean(speed_means)] * len(speed_std)
			y = speed_means
			print(cam_id)
			label='DUKEMTMC Cam' + str(cam_id)
			 # + ', {} trajectories'.format(len(traj_stat))
			if cam_id == 9:
				label = 'Town Centre Street'
			ax.scatter(x, y, label=label)
	ax.set_xlabel(r'Average pedestrian speed at specific scene ($m/s$)')
	ax.set_ylabel(r'Pedestrian speed ($m/s$)')
	ax.yaxis.grid(color='tab:gray', linestyle='--', linewidth=1., alpha=.3)
	plt.axvline(x=1., ls='--', color='k', lw=2.)
	ax.text(1.02, 0, r'Training speed')
	ax.legend(loc=2, framealpha=1.)
	# ax.spines['top'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['right'].set_visible(False)

	fig.tight_layout()
	fig.savefig('results/traj_stats.png', bbox_inches='tight')
	return None


def main():

	# plot_trajNums(trajNums_path)
	# plot_trajLens(trajLens_path)
	# plot_grids(grids_path)
	plot_speeds(speed_path)
	# plot_traj_stats(traj_stats_path)
	plt.show()


if __name__ == '__main__':

	main()


