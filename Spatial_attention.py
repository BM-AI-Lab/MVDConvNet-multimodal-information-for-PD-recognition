import numpy as np
import torch
from torch import nn


healthy_image = np.load("image_1.npy")
patients_image = np.load("image_2.npy")
print(healthy_image.shape, patients_image.shape)
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 压缩通道提取空间信息
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 经过卷积提取空间注意力权重
        x = torch.cat([max_out, avg_out], dim=1)
        out = self.conv1(x)
        # 输出非负
        out = self.sigmoid(out)
        return out

if __name__ == '__main__':
    SA = SpatialAttention()
    x = torch.from_numpy(healthy_image)
    x = x.to(torch.float32)
    y = torch.from_numpy(patients_image)
    y = y.to(torch.float32)
    x, y = x.unsqueeze(axis=1), y.unsqueeze(axis=1)
    SA1, SA2 = SA(x), SA(y)
    SA1, SA2 = SA1.detach().numpy(), SA2.detach().numpy()
    print(SA1, SA2.shape)
    np.save("image_1.npy", SA1)
    np.save("image_2.npy", SA2)
