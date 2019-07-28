# -*- coding: utf-8 -*-

import os
import pdb
import torch
import numpy as np
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from io_options.test_options import TestOptions
from models.data_preprocess import *


def quantify_error(ys_pred, ys):
	trans = ys[:, :-4]
	rotat = ys[:, -4:]
	trans_pred = ys_pred[:, :-4]
	rotat_pred = ys_pred[:, -4:]
	trans_err = np.linalg.norm(np.abs(trans - trans_pred), ord=2, axis=1)
	abs_distance = np.abs(np.sum(rotat * rotat_pred, axis=1))
	rotat_err = np.rad2deg(np.arccos(abs_distance))
	return np.mean(trans_err), np.mean(rotat_err)


def test(model,
         dataloader_test,
         device="cpu"):
	print('\nTesting...\n------------')

	with torch.no_grad():
		ys_pred, ys, Xs = [], [], []
		for step, (X, y, lens, intrin_mat) in enumerate(dataloader_test):
			X = Variable(torch.Tensor(X), requires_grad=False).to(device)
			y = Variable(torch.Tensor(y), requires_grad=False).to(device)
			y_pred = model(X, lens)
			ys_pred.append(y_pred.cpu().numpy())
			ys.append(y.cpu().numpy())
			Xs.append(X.cpu().numpy())
		ys_pred = np.concatenate(ys_pred, axis=0)
		ys = np.concatenate(ys, axis=0)[:, -5:]  # cut off the first 2 dimension
		Xs = np.concatenate(Xs, axis=0)
		t_err, r_err = quantify_error(ys_pred, ys)
	return t_err, r_err, ys_pred, ys


def main():

	# options
	test_opts = TestOptions().parse()
	data_test_file = os.path.join(test_opts.exp_dir, 'data_test.pkl')
	device = torch.device("cuda:"+str(test_opts.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
	test_epochs = np.arange(test_opts.num_epoch - 20, test_opts.num_epoch + 1, 5)

	dataloader_test = load_data(data_test_file, batch_size=test_opts.batch_size)
	best_epoch = -1
	t_rr_min = 1e10
	for test_epoch in test_epochs:
		model = torch.load(os.path.join(test_opts.checkpoints_dir, 'trained_model_epoch' + str(test_epoch) + '.pt')).eval().to(device)
		t_err, r_err, _, _ = test(model, dataloader_test, device=device)
		print('\nepoch: {}, translation error: {:.4f}m, rotation error {:.4f}'.format(test_epoch, t_err, r_err) + u'\u00b0')
		if t_rr_min > t_err:
			t_rr_min = t_err
			best_epoch = test_epoch

	# result of best model
	model = torch.load(os.path.join(test_opts.checkpoints_dir, 'trained_model_epoch' + str(best_epoch) + '.pt')).eval().to(device)
	t_err, r_err, ys_pred, ys = test(model, dataloader_test, device=device)
	print('\nground truth: {0}\nprediction: {1}'.format(ys[0], np.mean(ys_pred, axis=0)))
	print('translation error: {:.4f}m, rotation error {:.4f}'.format(t_err, r_err) + u'\u00b0')


if __name__ == '__main__':

	main()
