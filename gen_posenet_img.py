# -*- coding: utf-8 -*-

import pickle as pkl
import numpy as np
import torch
import shutil
import copy
import pdb
import os
from simulator.camera_parameters import *
from generate_data import *
from PIL import Image
from transforms3d.euler import euler2quat, quat2euler

# parse input arguments
parser = argparse.ArgumentParser(prog='PoseNet image generator arguments.')
parser.add_argument('--data_root', required=True, help='path to (training & test) trajectory data.')
args = parser.parse_args()
DATA_ROOT = args.data_root


def check_data_exist(data_root):
	data_train_path = os.path.join(data_root, 'data_train.pkl')
	data_test_path = os.path.join(data_root, 'data_test.pkl')
	if not os.path.exists(data_train_path):
		raise Exception('did not find training data')
	if not os.path.exists(data_test_path):
		raise Exception('did not find real test trajectory data')
	data_train = pkl.load(open(data_train_path, 'rb'))
	data_test = pkl.load(open(data_test_path, 'rb'))
	return data_train, data_test


def big_white_point(binary_img, pt, img_sz, window_sz=[11, 11]):

	pt_rela_x = window_sz[0] // 2
	pt_rela_y = window_sz[1] // 2
	for i in range(window_sz[0]):
		for j in range(window_sz[1]):
			x_rela = i - pt_rela_x
			y_rela = j - pt_rela_y
			x_abso = min(max(x_rela + pt[0], 0), img_sz[0]-1)
			y_abso = min(max(y_rela + pt[1], 0), img_sz[1]-1)
			binary_img[x_abso, y_abso] = 255
	return binary_img


def put_traj2img(traj, img_sz, rgb=True, img_dir=None):

	binary_img = np.zeros(img_sz, dtype=np.uint8)
	traj = traj.astype(int)
	for i in range(len(traj)):
		pt = traj[i]
		if 0 < pt[0] < img_sz[0] and 0 < pt[1] < img_sz[1]:
			binary_img[pt[0], pt[1]] = 255
			binary_img = big_white_point(binary_img, pt, img_sz, window_sz=[61, 61])
	binary_img = binary_img.T
	if rgb:
		binary_img = np.stack([binary_img, binary_img, binary_img], axis=-1)

	# # DEBUG: visualize image
	# fig = plt.figure()
	# ax = fig.add_subplot(111, aspect='auto')
	# ax.imshow(binary_img[:, :, 0], alpha=1., origin='lower')
	# plt.show()
	# pdb.set_trace()

	# # visualize
	# fig = plt.figure()
	# ax = fig.add_subplot(111, aspect='auto')
	# ax.imshow(binary_img[:, :, 0], alpha=1., origin='lower')
	# ax.plot(traj[:, 0], traj[:, 1], '-', markersize=1, c='w', alpha=1.)
	# # ax.scatter(traj[:, 0], traj[:, 1], marker='o', c='w', alpha=1.)
	# ax.get_xaxis().set_visible(False)
	# ax.get_yaxis().set_visible(False)
	# ax.axis('off')
	# fig.tight_layout()
	# if img_dir is not None:
	# 	fig.savefig(os.path.join(img_dir, 'posenet_binary_img.png'), bbox_inches='tight')
	# plt.show()
	# pdb.set_trace()

	return binary_img


def gen_posenet_imgs(data_dict, save_dir=DATA_ROOT, img_dir=None, save_img=True, save_label=True):

	img_save_dir = os.path.join(save_dir ,img_dir)
	if os.path.exists(img_save_dir):
		shutil.rmtree(img_save_dir)
	os.mkdir(img_save_dir)

	if save_img:
		trajs = data_dict['data']
		img_sz = data_dict['image_size']
		for i in range(len(trajs)):
			traj = trajs[i]
			img_arr = put_traj2img(traj, img_sz, rgb=True, img_dir=None)
			if img_dir is not None:
				img = Image.fromarray(img_arr, 'RGB')
				img_name = str(i) + '.png'
				img.save(os.path.join(img_save_dir, img_name))
				if i % 200 == 0:
					print('save image {}'.format(i))

	if save_label:
		labels = data_dict['label']
		label_file_name = 'dataset_' + img_dir + '.txt'
		label_file = open(os.path.join(save_dir, label_file_name), 'w')
		for i in range(len(labels)):
			posenet_label = labels[i]
			if img_dir is not None:
				img_name = str(i) + '.png'
				label_file.write(os.path.join(img_dir, img_name) + ' ' +
				                 ' '.join(map(str, posenet_label)) + '\n')
				if i % 200 == 0:
					print('write label for image {}'.format(i))
		label_file.close()
	return None


def main():

	# cam_par = CAMERA_TOWNCENTER
	data_train, data_test = check_data_exist(DATA_ROOT)
	gen_posenet_imgs(data_test,
	                 save_dir=DATA_ROOT,
	                 img_dir='test',
	                 save_img=True,
	                 save_label=True)
	gen_posenet_imgs(data_train,
	                 save_dir=DATA_ROOT,
	                 img_dir='train',
	                 save_img=True,
	                 save_label=True)


if __name__ == '__main__':

	main()
