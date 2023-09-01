from __future__ import print_function
from __future__ import absolute_import
from __future__ import with_statement
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import os
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


def takeSecond(elem):
    return elem[1]


# transpose maritx
class MyDataset(Dataset):
    def __init__(self, root_dir, datatxt, view_num, state, flod, numbers, random, transform=None,
                 target_transform=None):
        scaler = MinMaxScaler()
        path = os.path.join(root_dir, datatxt)
        data = sio.loadmat(path)
        # print(data)
        dataX = data['X']
        dataY = data["gt"]
        # dataY = data["Y"]
        images = []
        labels = []
        features = {}
        self.trains = {}
        self.vals = {}
        self.state = state
        # self.supervised=[]
        # self.supervised_label=[]
        # print(view_num)
        for i in range(view_num):
            features["{0}".format(i)] = dataX[0][i]
            features["{0}".format(i)] = np.transpose(features["{0}".format(i)])
            scaler.fit(features["{0}".format(i)])
            scaler.data_max_
            features["{0}".format(i)] = scaler.transform(features["{0}".format(i)])
            # print(features["{0}".format(i)].shape)
        for i in range(features["{0}".format(0)].shape[0]):
            label = dataY[i][0]
            image = features["{0}".format(0)][i]
            image = image.astype(np.double)
            image = torch.DoubleTensor(image)
            image = image.view(image.size(0), -1)
            image = torch.t(image)
            image = image.double()
            for j in range(1, view_num):
                a = features["{0}".format(j)][i].astype(np.double)
                a = torch.DoubleTensor(a)
                a = a.view(a.size(0), -1)
                a = torch.t(a)
                a = a.double()
                image = torch.cat((image, a), 1)
            images.append(image)
            labels.append(label)
        self.imgs = images
        # print(type(self.imgs))
        self.label = labels
        # print(type(self.label))
        self.transform = transform
        self.target_transform = target_transform
        self.train, self.test, self.train_label, self.test_label = train_test_split(self.imgs, self.label,
                                                                                    test_size=0.2, random_state=46,
                                                                                    stratify=self.label)
        _, self.supervised, _, self.supervised_label = train_test_split(self.train, self.train_label,
                                                                        test_size=0.1, random_state=46,
                                                                        stratify=self.train_label)

        # print(len(self.train),(self.train_label))
        kf = StratifiedKFold(n_splits=flod, shuffle=True, random_state=6)
        index = 0
        for train, val in kf.split(self.train, self.train_label):
            self.trains["{0}".format(index)] = train
            # print(train)
            self.vals["{0}".format(index)] = val
            # print(val)
            index += 1
        self.trainset = []
        self.trainset_label = []
        self.valset = []
        self.valset_label = []
        # print(self.trains["{0}".format(numbers)])
        # print(self.vals["{0}".format(numbers)])
        for i in self.trains["{0}".format(numbers)]:
            self.trainset.append(self.train[i])
            self.trainset_label.append(self.train_label[i])
        for i in self.vals["{0}".format(numbers)]:
            self.valset.append(self.train[i])
            self.valset_label.append(self.train_label[i])
        # print(self.trainset_label)
        # print(self.valset_label)
        # self.train.sort(key=takeSecond)
        # self.test.sort(key=takeSecond)

    def __getitem__(self, index):
        if self.state == 'train':
            img = self.trainset[index]
            label = self.trainset_label[index]
        elif self.state == 'test':
            img = self.test[index]
            label = self.test_label[index]
        elif self.state == 'val':
            img = self.valset[index]
            label = self.valset_label[index]
        elif self.state == 'train_test':
            img = self.train[index]
            label = self.train_label[index]
        elif self.state == 'all':
            img = self.imgs[index]
            label = self.label[index]
        elif self.state == "supervised":
            img = self.supervised[index]
            label = self.supervised_label[index]
        # is vector ,do not need transform
        return img, label, index

    def __len__(self):
        if self.state == 'train':
            return len(list(self.trainset))
        elif self.state == 'test':
            return len(list(self.test))
        elif self.state == 'val':
            return len(list(self.valset))
        elif self.state == 'train_test':
            return len(list(self.train))
        elif self.state == 'all':
            return len(list(self.imgs))
        elif self.state == "supervised":
            return len(list(self.supervised))