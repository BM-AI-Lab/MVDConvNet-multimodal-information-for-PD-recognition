# 编写人员：刘永灿
# 创建时间：2022/12/22 11:54
import re
import numpy as np
from torch import nn
from torchvision import models, transforms
import torch


# pretrained参数表示是否运用预训练模型
vgg_model = models.vgg19(pretrained=True)
# 这里重新定义最后一层，默认层输出的维度是1000，重新定义输出自己想要的维度。
new_classifier = torch.nn.Sequential(*list(vgg_model.children())[-1][:6])
# (6): Linear(in_features=4096, out_features=1000, bias=True)
new_classifier.add_module("Linear output", torch.nn.Linear(4096, 64))
vgg_model.features[0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg_model.features[2] = nn.ConvTranspose2d(16, 64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False)
vgg_model.features[34] = nn.ConvTranspose2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# new_classifier.add_module("Linear output", torch.nn.Linear(256, 1))
vgg_model.classifier = new_classifier
vgg_model.classifier[0] = nn.Linear(in_features=7*7, out_features=4096, bias=True)
# print(vgg_model)
trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


seq_info1 = np.load('gait.npy')


b = np.array([], dtype='float32')
x = 0
features = []
for i in np.nditer(seq_info1, order='C'):
    x += 1
    b = np.append(b, i)
    if x%1900 == 0:
        b = np.split(b, 19, 0)
        b = np.array(b)
        b = torch.Tensor(b)
        b = torch.unsqueeze(b, 0)
        vgg_model = vgg_model.eval()
        y = vgg_model(b).data.numpy()
        print(y.shape)
        features = np.append(features,y)
        b = np.array([], dtype='float32')
features = np.array(features)
print(features.shape)
features = features.reshape((13592, 128, 64))
np.save("gait.npy", features)