# -*- coding: utf-8 -*-
from .base_options import BaseOptions


class TestOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)
        # self.parser.add_argument('--num_epoch', type=int, default=100, help='number of training epochs')
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        self.isTrain = False