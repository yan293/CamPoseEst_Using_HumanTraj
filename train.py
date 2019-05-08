# -*- coding: utf-8 -*-

import os
import pdb
import torch
import argparse
import numpy as np
from torch import optim
from torchvision import transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models.data_preprocess import *
from io_options.train_options import TrainOptions
from models.trajnet import trajNetModelBidirect, Euc_Loss, Geo_Loss

# hyper-parameters
train_opts = TrainOptions().parse()
EPOCH = train_opts.num_epoch
BATCH_SIZE = train_opts.batch_size
INPUT_SIZE = train_opts.input_dim
OUTPUT_SIZE = train_opts.output_dim
LEARNING_RATE = train_opts.lr
DATA_TRAIN_FILE = os.path.join(train_opts.data_root, 'data_train.pkl')
MODEL_SAVE_PATH = train_opts.checkpoints_dir
# LOSS_CRITERION = Euc_Loss(beta=train_opts.beta)
LOSS_CRITERION = Geo_Loss()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = trajNetModelBidirect(INPUT_SIZE, OUTPUT_SIZE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def load_data(data_path, shuffle=True):
	''' Load data and generate data loader for network. '''
	data = TrajectoryCameraDataset(data_path, transform=transforms.Compose(
	                               [
	                               # RelativeTrajectory(),
	                               # NormalizeTo01(),
	                               ToTensor()
	                               ]))
	dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle,
	                        num_workers=4, collate_fn=pad_packed_collate_fn)
	return dataloader


def train(model,
          optimizer,
          dataloader_train,
          model_save_dir,
          dataloader_valid=None,
          record_log=False,
          device=DEVICE):

	# optimization setting
	train_log_path = os.path.join(model_save_dir, 'train_log')
	if not os.path.exists(train_log_path):
		os.mkdir(train_log_path)
	if os.listdir(train_log_path):
		for file in os.listdir(train_log_path):
			os.remove(os.path.join(train_log_path, file))
	writer = SummaryWriter(train_log_path)

	print('------------ Training -------------')
	for epoch in range(EPOCH):
		losses_train, losses_trans, losses_rotat = [], [], []
		for step, (X, y, lengths, intrin_mat) in enumerate(dataloader_train):

			# forward
			X = Variable(torch.Tensor(X), requires_grad=False)
			y = Variable(torch.Tensor(y), requires_grad=False)
			X, y = X.to(device), y.to(device)
			y_pred = model(X, lengths)
			intrin_mat = torch.Tensor(intrin_mat).to(device)
			loss_train, loss_trans, loss_rotat = LOSS_CRITERION(X, y, y_pred, intrin_mat)
			# loss_train, loss_trans, loss_rotat = LOSS_CRITERION(y[:, 2:], y_pred)
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
				losses_valid.append(LOSS_CRITERION(y_valid_pred, y_valid).item())
			loss_train = np.mean(losses_train)
			loss_valid = np.mean(losses_valid)
			if record_log:  # training log
				writer.add_scalar('loss/valid', loss_valid, epoch)
				writer.add_scalar('loss/train', loss_train, epoch)
			print('epoch: {} | training loss: {} | validation loss: {}'.format(epoch + 1, loss_train, loss_valid))
		else:
			print('epoch: {} | loss: {:.4f}, loss_t: {:.4f}, loss_r: {:.4f}'.format(epoch + 1, np.mean(losses_train), np.mean(losses_trans), np.mean(losses_rotat)))

		if (epoch+1)%5 == 0:
			print('model saved at epoch {}'.format(epoch+1))
			torch.save(model, os.path.join(model_save_dir, 'trained_model_epoch' + str(epoch+1) + '.pt'))
	print(y, '\n', y_pred)
	print('------------ End -------------')
	return model


def main():

	dataloader_train = load_data(DATA_TRAIN_FILE)
	trained_model = train(model,
	                      optimizer,
	                      dataloader_train,
	                      MODEL_SAVE_PATH,
	                      record_log=True,
	                      device=DEVICE)


if __name__ == '__main__':

	main()
