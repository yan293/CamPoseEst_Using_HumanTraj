# -*- coding: utf-8 -*-

import argparse


class SimulationOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--data_root', required=True, help='path to generated data')
		# self.parser.add_argument('--camera', required=True, help='camera used to generate synthetic data')
		self.parser.add_argument('--tz_range', type=list, default=[-3., 3.], help='range of translation z')
		self.parser.add_argument('--rx_range', type=list, default=[-5., 5.], help='range of rotation x1')
		self.parser.add_argument('--ry_range', type=list, default=[-10., 10.], help='range of pitch angle')
		self.parser.add_argument('--rz_range', type=list, default=[-5., 5.], help='range of rotation x2')
		self.parser.add_argument('--tz_inteval', type=float, default=0.4, help='step of translation z')
		self.parser.add_argument('--rx_inteval', type=float, default=1.5, help='step of rotation x')
		self.parser.add_argument('--ry_inteval', type=float, default=1.5, help='step of rotation y')
		self.parser.add_argument('--rz_inteval', type=float, default=1.5, help='step of rotation z')
		self.parser.add_argument('--traj_len', type=int, default=35, help='Length of each trajectory.')
		self.parser.add_argument('--traj_num', type=int, default=1, help='Number trajectories per camera.')
		self.parser.add_argument('--human_speed', type=float, default=1.4, help='Number trajectories per camera.')
		self.initialized = True

	def parse(self, data_root=None):
		if not self.initialized:
			self.initialize()
		if data_root is not None:
			self.opt = self.parser.parse_args(['--data_root', data_root])
		else:
			self.opt = self.parser.parse_args()

		# command line print
		args = vars(self.opt)
		print('------------ Simulation Options -------------')
		for k, v in sorted(args.items()):
			print('{}: {}'.format(str(k), str(v)))
		print('-------------- End ----------------')
		return self.opt
