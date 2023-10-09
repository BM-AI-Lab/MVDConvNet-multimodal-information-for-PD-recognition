# 编写人员：刘永灿
# 创建时间：2022/12/25 19:16
import re

import numpy as np
from sklearn.utils import shuffle




#获取融合数据
gait_data = np.load("gait.npy")
healthy_gait = shuffle(gait_data[0:4517, :, :])
patients_gait = shuffle(gait_data[4517:9034, :, :])
#print(healthy_gait.shape,healthy_gait, patients_gait.shape)
healthy_image = np.load("image_1.npy")
patients_image = np.load("image_2.npy")
healthy_image = np.repeat(healthy_image, 33, axis=0)
patients_image = np.repeat(patients_image, 37, axis=0)
healthy_image = shuffle(healthy_image[0:4517, :, :])
patients_image = shuffle(patients_image[0:4517, :, :])
#print(healthy_image.shape,patients_image.shape)
gait = np.concatenate((healthy_gait, patients_gait))
meander = np.concatenate((healthy_image, patients_image))
data_ = np.concatenate((gait, meander), axis=2)
print(data_.shape)   #(9034,128，128)

#标签
with open('D:\datasets\Ga13592class4label.txt', 'r') as f:
    seq_info2 = [
        re.split(r"[' '|'\t'|'\r'|'\n']+", line.strip()) for line in f.readlines()
    ]

labels_ = np.array(seq_info2, dtype=np.float32)

labels_ = labels_[0:9034, :]
data_ = data_.reshape((9034, 128*128))
data_ = data_.astype(np.float32)
con = np.concatenate((data_, labels_), axis=1)
con = shuffle(con)
data_ = con[:, 0:16384]
labels_ = con[:, -1]
print(con.shape, labels_, labels_.shape)
np.save("data_2.npy", data_)
np.save("labels_2.npy", labels_)
