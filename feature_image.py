# 编写人员：刘永灿
# 创建时间：2022/12/23 18:25
import os

import numpy as np
import torch
from torch import nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# import torch.utils.data as Data
# import re
# from sklearn.utils import shuffle
from torchvision import models, transforms
from PIL import Image
from Spatial_attention import SpatialAttention

# pretrained参数表示是否运用预训练模型
vgg_model = models.vgg19(pretrained=True)
# 这里重新定义最后一层，默认层输出的维度是1000，我这里重新定义可以输出自己想要的维度。
new_classifier = torch.nn.Sequential(*list(vgg_model.children())[-1][:6])
# (6): Linear(in_features=4096, out_features=1000, bias=True)
new_classifier.add_module("Linear output", torch.nn.Linear(4096, 8192))
vgg_model.features[0] = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg_model.features[2] = nn.ConvTranspose2d(16, 64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False)
vgg_model.features[34] = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# new_classifier.add_module("Linear output", torch.nn.Linear(256, 1))
vgg_model.classifier = new_classifier
vgg_model.classifier[0] = nn.Linear(in_features=12544, out_features=4096, bias=True)
print(vgg_model)
trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

SA = SpatialAttention()

i = 0
features = []
dir_path1 = "D:\datasets\Meander\HealthyMeander"
files = os.listdir(dir_path1)
for file in files:
    if i == 0:
            i += 1
    else:
        # print(file)
        file = os.path.join(dir_path1, file)
        im = Image.open(file)
        im = trans(im)
        im.unsqueeze_(dim=0)
        print(im.shape)
        SA1 = SA(im).detach()
        vgg_model = vgg_model.eval()
        y = vgg_model(im).data.numpy()
        # y = np.expand_dims(y,axis=1)
        # y = y.swapaxes(0,1)
        y = y.reshape((128,64))
        print(y.shape)
        features = np.append(features, y)
features = np.array(features).reshape((139,128,64))
# np.save("image_1.npy", features)
# features = np.expand_dims(features,axis=1)
# 扩充到4517



i = 0
patients = []
dir_path2 = "D:\datasets\Meander\PatientMeander"
files = os.listdir(dir_path2)
for file in files:
    if i == 0:
        i += 1
    else:
        # print(file)
        file = os.path.join(dir_path2, file)
        im = Image.open(file)
        im = trans(im)
        im.unsqueeze_(dim=0)
        SA2 = SA(im).detach()
        vgg_model = vgg_model.eval()
        z = vgg_model(im).data.numpy()
        z = z.reshape((128, 64))
        print(z.shape)
        patients = np.append(patients, z)
patients = np.array(patients).reshape((123, 128, 64))
# np.save("image_2.npy", patients)