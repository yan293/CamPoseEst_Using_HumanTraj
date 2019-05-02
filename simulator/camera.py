# -*- coding: utf-8 -*-

# from simulator.parameters import *
import numpy as np
import transforms3d
import cv2
import pdb


def euc_to_homo(pts_euc):
	'''
	convert euclidean coordinates to homogeneous coordinates,
	"pts_euc" Nx2, output Nx3
	'''
	pts_homo = np.hstack((pts_euc, np.ones([len(pts_euc), 1])))
	return pts_homo


def homo_to_euc(pts_homo):
	'''
	convert homogeneous coordinates to euclidean coordinates,
	"pts_homo" Nx3, output Nx2
	'''
	x_euc = pts_homo[:, 0] / pts_homo[:, -1]
	y_euc = pts_homo[:, 1] / pts_homo[:, -1]
	pts_euc = np.stack((x_euc, y_euc), axis=-1)
	return pts_euc


class IntrinsicCamera():

	def __init__(self, intrins=None):
		if intrins is None:
			self.fx = 1.
			self.fy = 1.
			self.ox = 0.
			self.oy = 0.
			self.s = 0.
		else:
			self.fx = intrins[0]  # focal length in 'x' direction (in pixel)
			self.fy = intrins[1]  # focal length in 'y' direction (in pixel)
			self.ox = intrins[2]  # 'x' coordinate of principle point (in pixel)
			self.oy = intrins[3]  # 'y' coordinate of principle point (in pixel)
			self.s = intrins[4]  # skew
		self.intrins = [self.fx, self.fy, self.ox, self.oy, self.s]

	def compute_intrin_mat(self):
		in_mat = np.array([[self.fx,   self.s,   self.ox],
			               [0,        self.fy,   self.oy],
			               [0,               0,        1]])
		return in_mat


class Camera(IntrinsicCamera):

	def __init__(self, cam_pars):
		if cam_pars['intrinsics'] is not None:
			super().__init__(intrins=cam_pars['intrinsics'])
		self.img_size = cam_pars['img_size']
		self.trans = cam_pars['translation']
		self.rotat_quat = cam_pars['rotation']
		self.extrins = list(cam_pars['translation']) + list(cam_pars['rotation'])
		self.distorts = cam_pars['distortion']

		self.in_mat = self.compute_intrin_mat()
		self.ex_mat = self.compute_extrin_mat()
		self.cam_mat = self.compute_cam_mat()

	def compute_extrin_mat(self):
		t_mat = np.hstack((np.eye(3), np.array([self.trans]).T))
		r_mat = transforms3d.quaternions.quat2mat(self.rotat_quat)
		ex_mat = r_mat.dot(t_mat)  # (3x4) extrinsic matrix
		return ex_mat

	def compute_extrin_mat_2(self):
		self.tx = self.trans[0]
		self.ty = self.trans[1]
		self.tz = self.trans[2]
		rotat_euler = transforms3d.euler.quat2euler(self.rotat_quat, axes='sxyz')
		self.rx = rotat_euler[0]
		self.ry = rotat_euler[1]
		self.rz = rotat_euler[2]
		t_mat = np.hstack((np.eye(3), np.array([[self.tx, self.ty, self.tz]]).T))
		r_mat = transforms3d.euler.euler2mat(self.rx, self.ry, self.rz, axes='sxyz')
		ex_mat = r_mat.dot(t_mat)  # (3x4) extrinsic matrix
		return ex_mat

	def compute_cam_mat(self):
		return self.in_mat.dot(self.ex_mat)

	def undistort(self, pts_2d):
		'''
		convert observed point coordinates to ideal
		(w/o distortion) point coordinates
		'''
		if self.distorts is None:
			undistort_pts_2d = pts_2d
		else:
			undistort_pts_2d = cv2.undistortPoints(np.expand_dims(pts_2d, 1),
			                                       self.in_mat,
			                                       self.distorts,
			                                       P=self.in_mat)
		undistort_pts_2d = undistort_pts_2d[:, 0, :]
		return undistort_pts_2d

	def project_ground2im(self, pts_3d_euc):
		'''
		project 3D points on ground plane (Z=0) to 2D image plane,
		"pts_3d_euc" are Nx2
		'''
		m_33 = self.cam_mat[:, [0, 1, -1]]
		pts_3d_homo = euc_to_homo(pts_3d_euc)
		pts_2d_homo = m_33.dot(pts_3d_homo.T).T
		pts_2d_euc = homo_to_euc(pts_2d_homo)
		return pts_2d_euc

	def project_im2ground(self, pts_2d_euc):
		'''
		project 2D points on image plane to 3D ground plane (Z=0),
		"pts_2d_euc" are Nx2
		'''
		m_33 = self.cam_mat[:, [0, 1, -1]]
		m_33_inv = np.linalg.inv(m_33)
		pts_2d_homo = euc_to_homo(pts_2d_euc)
		pts_3d_homo = m_33_inv.dot(pts_2d_homo.T).T
		pts_3d_euc = homo_to_euc(pts_3d_homo)
		return pts_3d_euc

	def compute_ground_fov(self):
		'''
		given camera parameters (intrinsic and extrinsic),
		compute ground plane FOV, output 4 vertices
		'''
		x_min, x_max = 0., self.img_size[0]
		y_min, y_max = 0., self.img_size[1]
		img_corners = np.array([[x_min, y_min],  # 4 corners of the image
		                        [x_max, y_min],
		                        [x_max, y_max],
		                        [x_min, y_max]])
		fov_corners = self.project_im2ground(img_corners)
		x_c, y_c = (x_min + x_max) / 2., (y_min + y_max) / 2.
		# pdb.set_trace()
		img_center = np.array([[x_c, y_c],
		                       [x_c + 10, y_c],
		                       [x_c, y_c + 10]])
		fov_center = self.project_im2ground(img_center)
		return fov_corners, fov_center

	def compute_visual_angle(self):
		fx = self.intrins[0]
		fy = self.intrins[1]
		img_sz_x = self.img_size[0]
		img_sz_y = self.img_size[1]
		theta_1 = np.rad2deg(np.arctan2(img_sz_x/2., fx)) * 2.
		theta_2 = np.rad2deg(np.arctan2(img_sz_y/2., fy)) * 2.
		return theta_1, theta_2
