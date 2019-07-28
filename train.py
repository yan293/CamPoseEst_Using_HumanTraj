# -*- coding: utf-8 -*-

import os
import pdb
import time
import torch
import random
import argparse
import numpy as np
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models.data_preprocess import *
from io_options.train_options import TrainOptions
from models.trajnet import *


def train(model,
          optimizer,
          loss_criterion,
          dataloader_train,
          model_save_dir,
          num_epoch=100,
          dataloader_valid=None,
          record_log=False,
          device="cpu"):

	# optimization setting
	train_log_path = os.path.join(model_save_dir, 'train_log')
	if not os.path.exists(train_log_path):
		os.mkdir(train_log_path)
	if os.listdir(train_log_path):
		for file in os.listdir(train_log_path):
			os.remove(os.path.join(train_log_path, file))
	writer = SummaryWriter(train_log_path)

	print('------------ Training -------------')
	time_start = time.time()
	for epoch in range(num_epoch):
		losses_train, losses_trans, losses_rotat = [], [], []
		for step, (X, y, lengths, intrin_mat) in enumerate(dataloader_train):

			# forward
			X = Variable(torch.Tensor(X), requires_grad=False)
			y = Variable(torch.Tensor(y), requires_grad=False)
			X, y = X.to(device), y.to(device)
			y_pred = model(X, lengths)
			loss_train, loss_trans, loss_rotat = loss_criterion(y[:, 2:], y_pred)
			losses_train.append(loss_train.item())
			losses_trans.append(loss_trans)
			losses_rotat.append(loss_rotat)

			# backward
			optimizer.zero_grad()
			loss_train.backward()
			optimizer.step()

		# validation
		if dataloader_valid is not None:
			losses_valid = []
			for _, (X_valid, y_valid, lens_valid) in enumerate(dataloader_valid):
				X_valid = Variable(torch.Tensor(X_valid), requires_grad=False).to(device)
				y_valid = Variable(torch.Tensor(y_valid), requires_grad=False).to(device)
				y_valid_pred = model(X_valid, lens_valid)
				losses_valid.append(loss_criterion(y_valid_pred, y_valid).item())
			loss_train = np.mean(losses_train)
			loss_valid = np.mean(losses_valid)
			if record_log:  # training log
				writer.add_scalar('loss/valid', loss_valid, epoch)
				writer.add_scalar('loss/train', loss_train, epoch)
			print('epoch: {} | training loss: {} | validation loss: {}'.format(epoch + 1, loss_train, loss_valid))
		else:
			print('epoch: {} | loss: {:.4f}, loss_t: {:.4f}, loss_r: {:.4f}'.format(epoch + 1, np.mean(losses_train), np.mean(losses_trans), np.mean(losses_rotat)))

		# save intermediate models
		if (epoch+1)%5 == 0:
			print('model saved at epoch {}'.format(epoch+1))
			torch.save(model, os.path.join(model_save_dir, 'trained_model_epoch' + str(epoch+1) + '.pt'))
	print('\nTotal training epoches: {}, total time cost: {} secs'.format(num_epoch, time.time()-time_start))
	print('------------ End -------------')
	return model


def main():

	# options
	train_opts = TrainOptions().parse()
	data_train_file = os.path.join(train_opts.exp_dir, 'data_train.pkl')
	loss_criterion = EucLoss(beta=train_opts.beta)
	device = torch.device("cuda:"+str(train_opts.gpu_ids[0]) if torch.cuda.is_available() else "cpu")

	# random seed
	manualSeed = 999
	print('Random Seed: {}'.format(manualSeed))
	random.seed(manualSeed)
	torch.manual_seed(manualSeed)

	# model and train
	model = trajNetModelBidirect(train_opts.input_dim,
	                             train_opts.output_dim).to(device)
	optimizer = optim.Adam(model.parameters(),
	                       lr=train_opts.lr)
	dataloader_train = load_data(data_train_file,
	                             batch_size=train_opts.batch_size)
	trained_model = train(model,
	                      optimizer,
	                      loss_criterion,
	                      dataloader_train,
	                      train_opts.checkpoints_dir,
	                      num_epoch=train_opts.num_epoch,
	                      record_log=True,
	                      device=device)


if __name__ == '__main__':

	main()
