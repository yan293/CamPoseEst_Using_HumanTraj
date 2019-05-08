# -*- coding: utf-8 -*-

import pdb
import torch
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset, DataLoader


class TrajectoryCameraDataset(Dataset):
	''' Trajectory Camera Extrinsic Parameter Dataset. '''

	def __init__(self, file_path, transform=None):

		self.file_path = file_path
		self.transform = transform
		self.raw_data = None


	def __len__(self):
		''' Size of data. '''
		if self.raw_data is None:
			self.raw_data = pkl.load(open(self.file_path, 'rb'))
		label = self.raw_data['label']
		return len(label)


	def __getitem__(self, idx):
		''' Get the 'idx'-th data and label. '''
		if self.raw_data is None:
			self.raw_data = pkl.load(open(self.file_path, 'rb'))

		data, label = self.raw_data['data'], self.raw_data['label']
		sample = {'trajectory': data[idx],
		          'camera_extinsics': label[idx],
		          'image_size': self.raw_data['image_size'],
		          'intrin_mat': self.raw_data['intrin_mat']
		          }
		if self.transform:
			sample = self.transform(sample)
		return sample


class ToTensor(object):
	''' convert numpy trajectory to pytorch tensor '''
	def __call__(self, sample):
		return {'trajectory': torch.from_numpy(sample['trajectory']),
				'camera_extinsics': torch.from_numpy(sample['camera_extinsics']),
				'image_size': torch.from_numpy(np.array(sample['image_size'])),
				'intrin_mat': torch.FloatTensor(np.array(sample['intrin_mat']))
				}


class RelativeTrajectory(object):
	''' calculate distance between points '''
	def __call__(self, sample):
		traj = sample['trajectory']
		traj_1, traj_2 = sample['trajectory'][:-1], sample['trajectory'][1:]
		traj_diff = traj_2 - traj_1
		traj[1:] = traj_diff
		# traj = traj_diff
		return {'trajectory': traj,
				'camera_extinsics': sample['camera_extinsics'],
				'image_size': sample['image_size'],
				'intrin_mat': torch.from_numpy(np.array(sample['intrin_mat']))
				}


class NormalizeTo01(object):
	''' Normalize trajectory coordinate to (0, 1) '''
	def __call__(self, sample):
		data, img_sz = sample['trajectory'], sample['image_size']
		data[:, 0] = data[:, 0] / float(img_sz[0])
		data[:, 1] = data[:, 1] / float(img_sz[1])
		return {'trajectory': data,
				'camera_extinsics': sample['camera_extinsics'],
				'image_size': sample['image_size'],
				'intrin_mat': torch.from_numpy(np.array(sample['intrin_mat']))
				}


def pad_packed_collate_fn(batch):
	'''
	Pad all other samples in a batch with 0, to the sample with longest length.
	Args:
      batch: (list of tuples) [(audio, target)].
          audio is a FloatTensor
          target is a LongTensor with a length of 8
    Output:
      packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
	'''
	batch.sort(key=lambda item: len(item['trajectory']), reverse=True)
	max_size = batch[0]['trajectory'].shape
	data, label, lengths = [], [], []
	for sample in batch:
		data_i = np.zeros(max_size)
		X_i, y_i = sample['trajectory'], sample['camera_extinsics']
		data_i[:len(X_i), :] = X_i
		data.append(data_i)
		label.append(y_i)
		lengths.append(len(X_i))
	data, label = np.stack(data, axis=0), np.stack(label, axis=0)
	intrin_mat = sample['intrin_mat']
	return data, label, lengths, intrin_mat
