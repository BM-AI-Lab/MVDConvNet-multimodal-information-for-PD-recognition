import re

import torch,  math
import torch.nn as nn
import numpy as np
from sklearn.utils import shuffle
import torch.nn.functional as F

with open('D:\datasets\Ga13592class4data.txt', 'r') as f:
    seq_info1 = [
        re.split(r"[' '|'\t'|'\r'|'\n']+", line.strip()) for line in f.readlines()
    ]

gait_data = np.array(seq_info1, dtype=np.float32)
gait_data = np.split(gait_data, 19, 1)
gait_data = np.array(gait_data)
gait_data = np.transpose(gait_data, axes=(1, 0, 2))
healthy_gait = shuffle(gait_data[0:4517, :, :])
patients_gait = shuffle(gait_data[4517:9034, :, :])
#print(healthy_gait.shape,healthy_gait, patients_gait.shape)
print(gait_data)
class MultiHead_SelfAttention(nn.Module):
    def __init__(self, dim, num_head):
        '''
        Args:
            dim: dimension for each time step
            num_head:num head for multi-head self-attention
        '''
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        self.qkv = nn.Linear(dim, dim * 3)  # extend the dimension for later spliting

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = q @ k.transpose(-1, -2) / math.sqrt(self.dim)
        att = att.softmax(dim=1)  # 将多个注意力矩阵合并为一个
        x = (att @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        return x


if __name__ == '__main__':
    num_head = 8
    x = gait_data[0:4517, :, :]
    y = gait_data[4517:9034, :, :]
    x = torch.from_numpy(x)
    x = x.to(torch.float32)
    MHSA1 = MultiHead_SelfAttention(64, num_head)
    y = torch.from_numpy(y)
    y = y.to(torch.float32)
    Normalization = torch.nn.BatchNorm2d(1)
    '''-------------------------time-------------------------------------'''
    '''time_att1, time_att2 = MHSA1(x), MHSA1(y)
    time_att1,time_att2 = time_att1.unsqueeze(axis=1), time_att2.unsqueeze(axis=1)
    time_att1, time_att2 = Normalization(time_att1), Normalization(time_att2)
    print(time_att1, time_att1.shape)
    print(time_att2, time_att2.shape)
    x1 = time_att1.detach().numpy()
    y1 = time_att2.detach().numpy()
    z1 = np.concatenate((x1, y1), axis=0)
    #np.save("gait with TA.npy", z1)
    print(z1.shape)
    '''
    '''---------------------force----------------------------'''
    '''x = torch.permute(x, (0, 2, 1))
    y = torch.permute(y, (0, 2, 1))
    print(x.shape)
    MHSA2 = MultiHead_SelfAttention(128, num_head)
    force_att1, force_att2 = MHSA2(x), MHSA2(y)
    force_att1, force_att2 = force_att1.unsqueeze(axis=1), force_att2.unsqueeze(axis=1)
    #force_att1, force_att2 = torch.nn.BatchNorm2d(force_att1), torch.nn.BatchNorm2d(force_att2)
    force_att1, force_att2 = Normalization(force_att1), Normalization(force_att2)
    force_att1, force_att2 = torch.permute(force_att1, (0, 1, 3, 2)), torch.permute(force_att2, (0, 1, 3, 2))
    print(force_att1, force_att1.shape)
    print(force_att2, force_att2.shape)
    x2 = force_att1.detach().numpy()
    y2 = force_att2.detach().numpy()
    z2 = np.concatenate((x2, y2), axis=0)
    np.save("gait with FA.npy", z2)
    '''



    TA, FA = np.load("gait with TA.npy"), np.load("gait with FA.npy")
    fuse = TA + FA
    fuse = torch.from_numpy(fuse)
    fuse = fuse.to(torch.float32)
    fuse = Normalization(fuse)
    fuse1 = F.normalize(fuse.float(), p=2, dim=2)
    fuse2 = F.normalize(fuse.float(), p=2, dim=3)
    fuse = (fuse1+fuse2)/2
    fuse = fuse.detach().numpy()
    fuse = np.squeeze(fuse, axis=1)
    print(fuse, fuse.shape)
    np.save("gait.npy", fuse)