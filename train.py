#!/usr/bin/python
# -*- coding: UTF-8 -*-

# pytorch -0.2.1
# python -3.6.2
# torchvision - 0.1.9
import csv
import re
import torch.utils.data as Data
import warnings

import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import os
import json
import math
# model
# from models.res2net.res2net import Res2Net, res2net50
from model import Res2Net, res2net50
# dataset
# import dataset.imagenet.dataset_imagenet
# loss
from loss import CrossEntropyLabelSmooth


##############################################################################################################

class FineTuner_CNN:
    def __init__(self, train_loader, test_loader, model):
        self.args = args
        self.learningrate = self.args.learning_rate
        self.learning_rate_decay = self.args.learning_rate_decay
        self.momentum = self.args.momentum
        self.weight_decay = self.args.weight_decay
        # self.train_path = self.args.train_path
        # self.test_path = self.args.test_path

        # imagenet
        self.train_data_loader = train_loader
        self.test_data_loader = test_loader

        self.model = model.cuda()
        # self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.criterion = nn.CrossEntropyLoss()

        self.accuracys1 = []
        self.accuracys5 = []

        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()

    ##############################################################################################################

    ''''------------------训练------------------'''
    def train(self, epoches=-1, batches=-1):
        epoch_i = -1
        if os.path.isfile("epoch_i"):
            epoch_i = torch.load("epoch_i")
            # print("epoch_i resume:", epoch_i)

            # self.model = torch.load("model_training_")
            # print("model_training resume:", self.model)

            self.accuracys1 = torch.load("accuracys1_trainning")
            self.accuracys5 = torch.load("accuracys5_trainning")
            # print("accuracys1_trainning resume:", self.accuracys1)
            # print("accuracys5_trainning resume:", self.accuracys5)

            self.test(0)

        accuracy1 = 0
        accuracy5 = 0
        for i in list(range(epoches)):
            print("Epoch: ", i)

            if i <= epoch_i:
                self.adjust_learning_rate(i)
                continue

            optimizer = optim.Adam(self.model.parameters(), lr=self.learningrate,
                                   weight_decay=self.weight_decay)

            for step, (batch, target) in enumerate(self.train_data_loader):
                batch = torch.unsqueeze(batch, 1)
                batch, target = Variable(batch.cuda()), Variable(target.cuda())  # Tensor->Variable
                output = self.model(batch)
                #output = torch.argmax(output, dim=1)
                #output = torch.unsqueeze(output, dim=1)
                target = torch.squeeze(target, 1)
                #print(output.size(), target.size())
                #target = torch.tensor(target, dtype=torch.int64)
                loss = self.criterion(output, target.long())
                loss.backward()

                optimizer.step()  # update parameters
                self.model.zero_grad()

                if step % self.args.print_freq == 0:
                    print("loss:", loss.data.cpu().numpy())

            self.test(epoch=i)

            # save the best model
        '''   if cor1 > accuracy1:
                torch.save(self.model, "model_training_m1")
                accuracy1 = cor1

            torch.save(i, "epoch_i")
            torch.save(self.model, "model_training_")
            torch.save(self.accuracys1, "accuracys1_trainning")
            torch.save(self.accuracys5, "accuracys5_trainning")

            self.adjust_learning_rate(i)
        '''



    ''''------------------测试------------------'''
    def test(self, flag=-1, epoch=-1):
        self.model.eval()

        print("Testing...")
        correct1 = 0
        p = 0
        r = 0
        f1 = 0
        j = 0
        total = 0

        with torch.no_grad():
            for i, (batch, target) in enumerate(self.test_data_loader):
                batch = torch.unsqueeze(batch, 1)
                #print(batch.size())
                batch, target = Variable(batch.cuda()), Variable(target.cuda())  # Tensor->Variable
                #print(batch.size())
                output = self.model(batch)
                Prob = F.sigmoid(output).cpu().numpy()
                target = torch.squeeze(target, dim=1)
                #print(target)
                predict, cor1, acc_ = accuracy(output, target)  # measure accuracy top1 and top5
                predict = predict.cpu().numpy()
                #print(predict)
                # 忽略警告
                warnings.filterwarnings("ignore")
                precision_ = metrics.precision_score(target.cpu().numpy(), predict)
                recall_ = metrics.recall_score(target.cpu().numpy(), predict)
                f1_score_ = metrics.f1_score(target.cpu().numpy(), predict)
                #print(precision, recall, f1_score)
                correct1 += cor1
                p += precision_
                r += recall_
                f1 += f1_score_
                total += target.size(0)


                # 保存指标
                out = open('MVD_.csv', 'a', newline='')
                csv_write = csv.writer(out, dialect='excel')
                #print(epoch)
                #if epoch == 39:
                if j == 0:
                    csv_write.writerow(
                        ['acc', 'precision', 'recall', 'f1-score', 'true_label', 'class0', 'class1', '预测结果'])
                for item in range(len(Prob)):
                    # print(lb_pred[j][0],)
                    csv_write.writerow(
                        ['', '', '', '', target.tolist()[item], Prob[item][0], Prob[item][1],
                        predict[item].tolist()])
                    #if item == len(Prob)-1:
                        #.writerow([acc_, precision_, recall_, f1_score_, '', '', '', ''])

                j += 1

        # print(total)

        if flag == -1:
            self.accuracys1.append(float(correct1) / total)

        print("Accuracy:", float(correct1) / total)
        acc = float(correct1) / total
        precision = p / j
        recall = r / j
        f1_score = f1 / j
        print("Precision: {:.3f}, Recall: {:.3f}, F1_score: {:.3f}".format(precision, recall, f1_score))

        '''
        # 保存指标
        out = open('MVD.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        if args.epochs == 0:
            csv_write.writerow(
                ['acc', 'precision', 'recall', 'f1-score', '测试标签', '类别0预测概率', '类别1预测概率', '预测结果'])
        for item in range(len(output)):
            # print(lb_pred[j][0],)
            csv_write.writerow(['', '', '', '', target.cpu().numpy().tolist()[item], output[item][0], output[item][1],
                                 predict[item].tolist()])

            csv_write.writerow([acc, precision, recall, f1_score, '', '', '', ''])
        '''



        self.model.train()

        return float(correct1) / total

    def adjust_learning_rate(self, epoch):
        # manually
        if self.args.learning_rate_decay == 0:
            # imagenet
            if epoch in [30, 60, 90]:
                self.learningrate = self.learningrate / 10
        # exponentially
        elif self.args.learning_rate_decay == 1:
            num_epochs = 60
            lr_start = 0.01
            # print("lr_start = "+str(self.lr_start))
            lr_fin = 0.0001
            # print("lr_fin = "+str(self.lr_fin))
            lr_decay = (lr_fin / lr_start) ** (1. / num_epochs)
            # print("lr_decay = "+str(self.lr_decay))

            self.learningrate = self.learningrate * lr_decay
        # print("self.learningrate", self.learningrate)


##############################################################################################################

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""

    batch_size = target.size(0)

    predict = torch.max(output, dim=1)[1]
    pred = predict.t()
    correct = torch.eq(pred, target.cuda()).sum().item()
    acc = correct / 32

    return predict, correct, acc


##############################################################################################################

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CNN Training')

    parser.add_argument('--arch', '--a', default='ResNet', help='model architecture: (default: ResNet)')
    parser.add_argument('--epochs', type=int, default=40, help='number of total epochs to run')
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.00001, help='initial learning rate')
    parser.add_argument('--learning_rate_decay', '--lr_decay', type=int, default=0,
                        help='maually[0] or exponentially[1] decaying learning rate')
    parser.add_argument('--momentum', '--mm', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', type=float, default=0.5, help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', '--p', type=int, default=100, help='print frequency (default:20)')
    # imagenet
    '''parser.add_argument('--train_path', type=str,
                        default='D:\datasets\Meander\\train',
                        help='train dataset path')
    parser.add_argument('--test_path', type=str,
                        default='D:\datasets\Meander\\train',
                        help='test dataset path')
    '''
    parser.add_argument("--parallel", type=int, default=1)
    parser.set_defaults(train=True)
    args = parser.parse_args()

    return args


##############################################################################################################

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1,0"

    args = get_args()
    print("args:", args)

    model = res2net50()
    # model = torch.load("model")
    # torch.save(model, "model")
    # print("model_training:", model)

    if args.parallel == 1:
        model = torch.nn.DataParallel(model).cuda()



    # 预处理
    def normalization(data, label):

        mm_x = MinMaxScaler()  # 导入sklearn的预处理容器
        mm_y = MinMaxScaler()
        # data=data.values    # 将pd的系列格式转换为np的数组格式
        # label=label.values
        data = mm_x.fit_transform(data)  # 对数据和标签进行归一化等处理
        label = mm_y.fit_transform(label)
        return data, label


    # 数据分离
    def split_data(x, y, split_ratio):

        train_size = int(len(y) * split_ratio)
        test_size = len(y) - train_size

        x_data = Variable(torch.Tensor(np.array(x)))
        y_data = Variable(torch.Tensor(np.array(y)))

        x_train = Variable(torch.Tensor(np.array(x[0:train_size])))
        y_train = Variable(torch.Tensor(np.array(y[0:train_size])))
        y_test = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
        x_test = Variable(torch.Tensor(np.array(x[train_size:len(x)])))

        print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
              .format(x_data.shape, y_data.shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape))

        return x_data, y_data, x_train, y_train, x_test, y_test


    def data_generator(x_train, y_train, x_test, y_test, batch_size):

        # num_epochs = n_iters / (len(x_train) / batch_size)  # n_iters代表一次迭代
        # num_epochs = int(num_epochs)
        train_dataset = Data.TensorDataset(x_train, y_train)
        test_dataset = Data.TensorDataset(x_test, y_test)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                                   drop_last=True)  # 加载数据集,使数据集可迭代
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                                  drop_last=True)

        return test_dataset, train_loader, test_loader


    data_ = np.load("data_2.npy")
    label_ = np.load("labels_2.npy")
    label_ = np.expand_dims(label_, axis=1)
    data_, label_ = normalization(data_, label_)
    data_ = data_.reshape((9034, 128, 128))
    print(data_, label_, data_.shape, label_.shape)
    x_data, y_data, x_train, y_train, x_test, y_test = split_data(data_, label_, split_ratio=0.7)
    test_dataset, train_loader, test_loader = data_generator(x_train, y_train, x_test, y_test, batch_size=32)
    test_num = len(test_dataset)

    fine_tuner = FineTuner_CNN(train_loader, test_loader, model)

    fine_tuner.train(epoches=args.epochs)
    fine_tuner.test()
    # torch.save(fine_tuner.model, "model_training_final")
    # print("model_training_final:", fine_tuner.model)
