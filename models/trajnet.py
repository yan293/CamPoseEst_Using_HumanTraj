# -*- coding: utf-8 -*-

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import transforms3d
import numpy as np

# single-branch single-directional model
class trajNetModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=64):
        super(trajNetModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=self.hidden_dim,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=False)
        self.linear_1 = nn.Linear(self.hidden_dim, 128)
        self.linear_2 = nn.Linear(128, 1024)
        self.linear_3 = nn.Linear(1024, 512)
        self.linear_4 = nn.Linear(512, 256)
        self.output = nn.Linear(256, output_size)


    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        _, (h_T, _) = self.rnn(packed)
        rnn_out = h_T.squeeze()
        hidden = F.relu(self.linear_1(rnn_out))
        hidden = F.relu(self.linear_2(hidden))
        hidden = F.relu(self.linear_3(hidden))
        hidden = F.relu(self.linear_4(hidden))
        output = self.output(hidden)
        return output


# bidirectional model with branches
class trajNetModelBidirect(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=64):
        super(trajNetModelBidirect, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirect = True
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=self.hidden_dim,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=self.bidirect)

        # shared layers
        self.linear_1 = nn.Linear(self.hidden_dim*2, 256) # bidrection, so x2
        self.linear_2 = nn.Linear(256, 1024)
        self.linear_3 = nn.Linear(1024, 512)

        # branch layers
        self.branch_t_1 = nn.Linear(512, 256)
        self.branch_t_2 = nn.Linear(256, 128)
        self.branch_t_3 = nn.Linear(128, 1)

        self.branch_r_1 = nn.Linear(512, 256)
        self.branch_r_2 = nn.Linear(256, 128)
        self.branch_r_3 = nn.Linear(128, 4)


    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        self.rnn.flatten_parameters()
        _, (h_T, _) = self.rnn(packed)
        rnn_out = h_T.squeeze()
        if self.bidirect:
            # r_1, r_2 = rnn_out[0], rnn_out[1]
            rnn_out = torch.cat((rnn_out[0], rnn_out[1]), 1)

        # shared layers
        hidden = F.relu(self.linear_1(rnn_out))
        hidden = F.relu(self.linear_2(hidden))
        hidden = F.relu(self.linear_3(hidden))

        # branch layers
        hidden_t = F.relu(self.branch_t_1(hidden))
        hidden_t = F.relu(self.branch_t_2(hidden_t))
        trans = self.branch_t_3(hidden_t)

        hidden_r = F.relu(self.branch_r_1(hidden))
        hidden_r = F.relu(self.branch_r_2(hidden_r))
        rotat = F.normalize(self.branch_r_3(hidden_r), p=2, dim=1)

        output = torch.cat((trans, rotat), dim=1)

        return output


class trajNetModelBidirectBN(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=64):
        super(trajNetModelBidirectBN, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirect = True
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=self.hidden_dim,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=self.bidirect)

        # shared layers
        self.linear_1 = nn.Linear(self.hidden_dim*2, 256) # bidrection, so x2
        self.linear_2 = nn.Linear(256, 1024)
        self.linear_3 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=1024)
        self.bn2 = nn.BatchNorm1d(num_features=512)

        # branch layers
        self.branch_t = nn.Linear(512, 1)
        self.branch_r = nn.Linear(512, 4)


    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        self.rnn.flatten_parameters()
        _, (h_T, _) = self.rnn(packed)
        rnn_out = h_T.squeeze()
        if self.bidirect:
            # r_1, r_2 = rnn_out[0], rnn_out[1]
            rnn_out = torch.cat((rnn_out[0], rnn_out[1]), 1)

        # shared layers
        hidden = F.relu(self.bn1(self.linear_1(rnn_out)))
        hidden = F.relu(self.bn2(self.linear_2(hidden)))
        hidden = F.relu(self.bn3(self.linear_3(hidden)))

        # branch layers
        trans = self.branch_t(hidden)
        rotat = self.branch_r(hidden)
        output = torch.cat((trans, rotat), dim=1)

        return output


class Euc_Loss(torch.nn.Module):

    def __init__(self, beta=500):
        super(Euc_Loss,self).__init__()
        self.beta = beta


    def forward(self, y, y_pred):
        t_pred = y_pred[:, :-4]
        r_pred = y_pred[:, -4:]
        # r_pred = F.normalize(r_pred, p=2, dim=1)
        loss_t = F.l1_loss(t_pred, y[:, :-4])
        loss_r = F.l1_loss(r_pred, y[:, -4:])
        loss = loss_t + self.beta * loss_r
        return loss, loss_t.item(), loss_r.item()


class Geo_Loss(torch.nn.Module):

    def __init__(self):
        super(Geo_Loss, self).__init__()
        # self.intrin_mat = intrin_mat


    def forward(self, X, y, y_pred, intrin_mat):
        if y.shape[1] < 7:
            assert("extrinsics label must be a dimensions of Nx7")
        y_pred = torch.cat((y[:, :2], y_pred),1)
        t_pred_batch = y_pred[:, :-4]
        r_pred_quat_batch = F.normalize(y_pred[:, -4:], p=2, dim=1)
        t_label_batch = y[:, :-4]
        r_label_batch = y[:, -4:]

        # compute camera matrix
        ex_mat_batch_pred = self.compute_extrin_mat(t_pred_batch, r_pred_quat_batch)
        ex_mat_batch = self.compute_extrin_mat(t_label_batch, r_label_batch) # ALL MOST SURE CORRECT UNTIL THIS
        in_mat_batch = intrin_mat.repeat(len(ex_mat_batch), 1, 1)
        cam_mat_batch_pred = in_mat_batch.bmm(ex_mat_batch_pred)
        cam_mat_batch = in_mat_batch.bmm(ex_mat_batch)

        # project from image to ground
        traj_3d_batch_pred = self.project_im2ground(cam_mat_batch_pred, X)
        traj_3d_batch = self.project_im2ground(cam_mat_batch, X)

        # euclidean loss of trajectory
        traj_diff = traj_3d_batch_pred - traj_3d_batch
        loss = torch.norm(traj_diff, p=2, dim=-1).sum(dim=-1).mean()
        loss_t = F.l1_loss(y_pred[:, :-4], y[:, :-4])
        loss_r = F.l1_loss(y_pred[:, -4:], y[:, -4:])

        return loss, loss_t.item(), loss_r.item()


    def project_im2ground(self, cam_mat_batch, traj_2d_batch):
        '''
        project trajectory from img to ground, implemented in pytorch (cuda)
        '''
        device = "cpu"
        if traj_2d_batch.is_cuda:
            device = "cuda:" + str(traj_2d_batch.get_device())

        max_traj_len = traj_2d_batch.shape[1]
        traj_3d_batch = torch.zeros(traj_2d_batch.shape).to(device)

        for i in range(len(cam_mat_batch)):
            cam_mat = cam_mat_batch[i][:, [0, 1, 3]]
            cam_mat_pinv = torch.pinverse(cam_mat)
            traj_2d = traj_2d_batch[i]

            # unpadding
            idx_non_0 = torch.mul(traj_2d[:,0]>0, traj_2d[:,1]>0)
            traj_2d = traj_2d[idx_non_0]

            # homogeneous projection
            traj_2d_homo = torch.cat((traj_2d, torch.ones(len(traj_2d), 1).to(device)), 1)
            traj_3d_homo = cam_mat_pinv.mm(traj_2d_homo.t())
            traj_3d_homo = traj_3d_homo / traj_3d_homo[-1,:].repeat(3,1)
            traj_3d = traj_3d_homo[:-1, :].t()

            traj_3d_batch[i, :len(traj_3d), :] = traj_3d

        return traj_3d_batch


    def compute_extrin_mat(self, t_vec_batch, r_quad_batch):

        r_mat_batch = self.quat2rmat(r_quad_batch)
        rt_vec_batch = r_mat_batch.bmm(t_vec_batch.unsqueeze(-1))
        ex_mat_batch = torch.cat((r_mat_batch, rt_vec_batch), -1)

        # # DEBUG BY YAN
        # diff = []
        # for i in range(len(r_mat_batch)):
        #     trans = t_vec_batch[i]
        #     rotat_quat = r_quad_batch[i]
        #     ex_mat = self.extrin_mat_2(trans, rotat_quat)
        #     aa = ex_mat_batch[i].data.cpu().numpy()
        #     # print(np.max(np.abs(ex_mat - aa)))
        #     diff.append(np.max(np.abs(ex_mat - aa)))
        # print(np.max(diff))
        # pdb.set_trace()

        return ex_mat_batch


    def trans2tmat(self, trans_batch):
        t_vec_batch = torch.zeros(len(trans_batch), 3)
        if t_vec_batch.shape[1] == 1:
            t_vec_batch[:, -1] = trans_batch.squeeze()
        elif t_vec_batch.shape[1] == 3:
            t_vec_batch[:, -1] = trans_batch
        else:
            assert("translation vector dimension is wrong!")


    def quat2rmat(self, quat_batch, order='xyz'): # here 'zyx' equals 'xyz' in transforms3d.euler.quat2euler()
        '''
        Convert quaternion to rotation matrix
        '''
        device = "cpu"
        if quat_batch.is_cuda:
            device = "cuda:" + str(quat_batch.get_device())

        r_euler_batch = self.quat2euler(quat_batch, order)
        r_mat_batch = torch.zeros(len(quat_batch), 3, 3).to(device)

        # euler angle is correct, but matrix is not
        for i in range(len(r_euler_batch)):
            r_euler = r_euler_batch[i]
            r_1 = torch.FloatTensor([[1,                0,               0],
                                     [0,  torch.cos(r_euler[0]), torch.sin(r_euler[0])],
                                     [0, -torch.sin(r_euler[0]), torch.cos(r_euler[0])]])
            r_2 = torch.FloatTensor([[torch.cos(r_euler[1]), 0, -torch.sin(r_euler[1])],
                                     [0,               1,                0],
                                     [torch.sin(r_euler[1]), 0, torch.cos(r_euler[1])]])
            r_3 = torch.FloatTensor([[torch.cos(r_euler[2]),  torch.sin(r_euler[2]), 0],
                                     [-torch.sin(r_euler[2]), torch.cos(r_euler[2]), 0],
                                     [0,                              0, 1]])
            r_mat = r_1.mm(r_2.mm(r_3)).t()
            r_mat_batch[i, :, :] = r_mat

            # # DEBUG BY YAN, RESULT CONSISTANT WITH "transforms3d"
            # aa = r_mat.data.numpy()
            # bb = transforms3d.quaternions.quat2mat(quat_batch[i].data.cpu().numpy())
            # print(np.max(aa - bb))
            # pdb.set_trace()
        return r_mat_batch


    def quat2euler(self, quat, order, epsilon=0):
        """
        Convert quaternion(s) quat to Euler angles.
        Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert quat.shape[-1] == 4

        original_shape = list(quat.shape)
        original_shape[-1] = 3
        quat = quat.view(-1, 4)

        q0 = quat[:, 0]
        q1 = quat[:, 1]
        q2 = quat[:, 2]
        q3 = quat[:, 3]

        if order == 'zyx': # original 'xyz'
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
        elif order == 'xzy': # original 'yzx'
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
        elif order == 'yxz': # original 'zxy'
            x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
        elif order == 'yzx': # original 'xzy'
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
        elif order == 'zxy': # original 'yxz'
            x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
            y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        elif order == 'xyz': # original 'zyx'
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
            z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
        else:
            raise
        return torch.stack((x, y, z), dim=1).view(original_shape)


    # DEBUG BY YAN
    def extrin_mat_2(self, trans, rotat_quat):
        t_mat = np.hstack((np.eye(3), np.array([trans.data.cpu().numpy()]).T))
        r_mat = transforms3d.quaternions.quat2mat(rotat_quat.data.cpu().numpy())
        ex_mat = r_mat.dot(t_mat)  # (3x4) extrinsic matrix
        return ex_mat
