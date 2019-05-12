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

# hyper-parameters
test_opts = TestOptions().parse()
BATCH_SIZE = test_opts.batch_size
DATA_TEST_FILE = os.path.join(test_opts.data_root, 'data_test.pkl')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEST_EPOCHS = np.arange(60, 100 + 1, 5)


def quantify_error(ys_pred, ys):
	trans = ys[:, :-4]
	rotat = ys[:, -4:]
	trans_pred = ys_pred[:, :-4]
	rotat_pred = ys_pred[:, -4:]
	trans_err = np.linalg.norm(np.abs(trans - trans_pred), ord=2, axis=1)
	abs_distance = np.abs(np.sum(rotat * rotat_pred, axis=1))
	rotat_err = np.rad2deg(np.arccos(abs_distance))
	return np.mean(trans_err), np.mean(rotat_err)


# test
def test(model,
         dataloader_test,
         device=DEVICE):
	print('\n3) Testing...\n------------')

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
		ys = np.concatenate(ys, axis=0)[:, -5:]
		Xs = np.concatenate(Xs, axis=0)
		t_err, r_err = quantify_error(ys_pred, ys)

		# prediction
		y_pred_mean = np.mean(ys_pred, axis=0)
		y_pred_mean[-3:] = np.degrees(y_pred_mean[-3:])
		y_real = np.mean(ys, axis=0)
		y_real[-3:] = np.degrees(y_real[-3:])
	return t_err, r_err


def main():

	dataloader_test = load_data(DATA_TEST_FILE, batch_size=BATCH_SIZE)
	for test_epoch in TEST_EPOCHS:
		model = torch.load(os.path.join(test_opts.checkpoints_dir, 'trained_model_epoch' + str(test_epoch) + '.pt')).eval().to(DEVICE)
		t_err, r_err = test(model, dataloader_test, device=DEVICE)
		print('\nepoch: {}, translation error: {:.4f}m, rotation error {:.4f}'.format(test_epoch, t_err, r_err) + u'\u00b0')


if __name__ == '__main__':

	main()
