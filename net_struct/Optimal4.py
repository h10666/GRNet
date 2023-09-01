import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

__all__=['Optimal4']
class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
class Optimal4(nn.Module):
    def __init__(self, total_num,first,middle,end,num_classes):
        super(Optimal4, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(total_num, first),
            nn.BatchNorm1d(first),
            nn.LeakyReLU(),
            
            nn.Linear(first, middle),
            nn.BatchNorm1d(middle),
            nn.LeakyReLU()
            
            #nn.Linear(512, 512),
            #nn.LeakyReLU(),
            #nn.BatchNorm1d(512),

        )
        # self.decoder = nn.Sequential(
        #     nn.Linear(middle, first),
        #     nn.BatchNorm1d(first),
        #     nn.LeakyReLU(),
        #     nn.Linear(first, total_num),
        #     nn.Sigmoid(),
        # )

        self.attention1 = nn.Sequential(
            nn.Linear(middle,middle),
            # nn.Linear(middle,middle),
            nn.Sigmoid(),
        )
        self.attention2 = nn.Sequential(
            nn.Linear(middle, middle),
            # nn.Linear(middle,middle),
            nn.Sigmoid(),
        )
        self.attention3 = nn.Sequential(
            nn.Linear(middle, middle),
            # nn.Linear(middle,middle),
            nn.Sigmoid(),
        )
        self.attention4 = nn.Sequential(
            nn.Linear(middle, middle),
            # nn.Linear(middle,middle),
            nn.Sigmoid(),
        )
        self.Globalfeature = nn.Sequential(
            #nn.Linear(middle, 256),
            #nn.LeakyReLU(),
            #nn.BatchNorm1d(256),
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
        # decoded = self.decoder(encoded)
        #print("feature",encoded)
        # print(encoded.shape)
        attention1 = self.attention1(encoded)
        attention2 = self.attention2(encoded)
        attention3 = self.attention3(encoded)
        attention4 = self.attention4(encoded)
        # print("attention",attention1.shape)
        f1 = torch.mul(encoded, attention1)
        #f1=self.l2norm(f1)
        f2 = torch.mul(encoded, attention2)
        #f2=self.l2norm(f2)
        f3 = torch.mul(encoded, attention3)
        #f3=self.l2norm(f3)
        f4 = torch.mul(encoded, attention4)
        #f4=self.l2norm(f4)
        feature1 = self.Globalfeature(f1)
        #feature1=self.l2norm(feature1)
        feature2 = self.Globalfeature(f2)
        #feature2=self.l2norm(feature2)
        feature3 = self.Globalfeature(f3)
        #feature3=self.l2norm(feature3)
        feature4 = self.Globalfeature(f4)
        #feature4=self.l2norm(feature4)
        feature = torch.cat((feature1, feature2), 1)
        feature = torch.cat((feature, feature3), 1)
        feature = torch.cat((feature, feature4), 1)
        
        # print("feature_shape",feature.shape)

        
        return [attention1, attention2, attention3, attention4], [feature1, feature2, feature3, feature4], feature


