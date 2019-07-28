# -*- coding: utf-8 -*-
import argparse
import torch
import pdb
import os


class BaseOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False


    def initialize(self):
        self.parser.add_argument('--exp_dir', required=True, help='path to training data')
        self.parser.add_argument('--batch_size', type=int, default=1024, help='input batch size')
        self.parser.add_argument('--num_epoch', type=int, default=60, help='number of training epochs')
        self.parser.add_argument('--input_dim', type=int, default=2, help='dimension of input (2-D trajectory).')
        self.parser.add_argument('--output_dim', type=int, default=5, help='dimension of output (6-D or 4-D DoF).')
        self.parser.add_argument('--lstm_hidden_size', type=int, default=128, help='hidden size of the LSTM layer')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--model', type=str, default='trajnet', help='chooses which dnn model to use.')
        self.parser.add_argument('--dataLoaderThreads', default=8, type=int, help='# threads for loading data')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images batches in order, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=224,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--no_flip', action='store_true', default=True, help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--seed', type=int, default=0, help='initial random seed for deterministic results')
        self.parser.add_argument('--beta', type=float, default=100, help='beta factor used in posenet.')
        # self.parser.add_argument('--checkpoints_dir', type=str, default='./experiments/checkpoints', help='models are saved here')
        # self.parser.add_argument('--resize_or_crop', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        # self.parser.add_argument('--exp_name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.initialized = True


    def parse(self, exp_dir=None):

        # initialize hyper-settings
        if not self.initialized:
            self.initialize()
        if exp_dir is not None:
            self.opt = self.parser.parse_args(['--exp_dir', exp_dir])
        else:
            self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        # set gpu
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        if len(self.opt.gpu_ids) > 0:
            torch.device("cuda:" + str(self.opt.gpu_ids[0]))

        # print out settings on command window
        args = vars(self.opt)
        if self.opt.isTrain:
            phase = 'Train'
        else:
            phase = 'Test'
        print('------------ ' + phase + ' Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # -- save to the disk
        if not os.path.exists(self.opt.exp_dir):
            os.mkdir(self.opt.exp_dir)
        self.opt.checkpoints_dir = os.path.join(self.opt.exp_dir, 'checkpoints')
        if not os.path.exists(self.opt.checkpoints_dir):
            os.mkdir(self.opt.checkpoints_dir)

        file_name = os.path.join(self.opt.exp_dir, 'opt_'+self.opt.phase+'.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
