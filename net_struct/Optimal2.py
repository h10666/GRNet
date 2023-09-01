import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

__all__=['Optimal2']
class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
class Optimal2(nn.Module):
    def __init__(self, total_num,first,middle,end, num_classes):
        super(Optimal2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(total_num, first),
            nn.BatchNorm1d(first),
            nn.LeakyReLU(),

            nn.Linear(first, middle),
            nn.BatchNorm1d(middle),
            nn.LeakyReLU(),

            #nn.Linear(middle, middle),
            #nn.LeakyReLU(),
            #nn.BatchNorm1d(middle),

        )
        self.attention1 = nn.Sequential(
            nn.Linear(middle,middle),
            nn.Sigmoid(),
        )
        self.attention2 = nn.Sequential(
            nn.Linear(middle,middle),
            nn.Sigmoid(),
        )
        self.Globalfeature = nn.Sequential(
            #nn.Linear(middle, 192),
            #nn.LeakyReLU(),
            #nn.BatchNorm1d(192),
            nn.Linear(middle, end),
            nn.BatchNorm1d(end),
            #nn.LeakyReLU(),
            
        )
        #self.l2norm = Normalize(2)

    def forward(self, x):
        #print("x",x.shape)
        x = x.view(x.size(0), -1)
        #print("x",x.shape)
        encoded = self.encoder(x)
        #print("feature",encoded)
        # print(encoded.shape)
        attention1 = self.attention1(encoded)
        attention2 = self.attention2(encoded)
        # print("attention",attention1.shape)
        f1 = torch.mul(encoded, attention1)
        f2 = torch.mul(encoded, attention2)
        feature1 = self.Globalfeature(f1)
        #feature1=self.l2norm(feature1)
        feature2 = self.Globalfeature(f2)
        #feature2=self.l2norm(feature2)
        feature = torch.cat((feature1, feature2), 1)
        # print("feature_shape",feature.shape)


        return [attention1, attention2],[feature1, feature2], feature


