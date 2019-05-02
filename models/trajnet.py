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
        super(Geo_Loss,self).__init__()
        # self.intrin_mat = intrin_mat

    def forward(self, x, y, y_pred):
        # compute ground truth 3D trajectory
        t_pred = y_pred[:, :-4]
        r_pred_quat = F.normalize(y_pred[:, -4:], p=2, dim=1)
        r_mat = self.quat2mat(r_pred_quat)
        t_mat = torch.stack((torch.eye(3), t_pred.unsqueeze(-1)), 1)
        pdb.set_trace()
        r_mat = 12
        return None

    def quat2mat(self, quat, order='zyx'): # here 'zyx' equals 'xyz' in transforms3d.euler.quat2euler()
        '''
        Convert quaternion to rotation matrix
        '''
        r_euler_batch = self.quat2euler(quat, order)

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
            bb=transforms3d.quaternions.quat2mat(quat[0].data.numpy())
            print(r_mat.data.numpy() - bb)
            aa=np.rad2deg(transforms3d.euler.quat2euler(quat[i].data.numpy(), axes='sxyz'))
            cc=np.rad2deg(r_euler_batch[i].data.numpy())
            print(aa, '\n', cc)
            pdb.set_trace()

            # 0428 night, transforms3d and hard-coded quat2mat do not match, debug...

        return None

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

        if order == 'xyz':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
        elif order == 'yzx':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
        elif order == 'zxy':
            x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
        elif order == 'xzy':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
        elif order == 'yxz':
            x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
            y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        elif order == 'zyx':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
            z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
        else:
            raise
        return torch.stack((x, y, z), dim=1).view(original_shape)
