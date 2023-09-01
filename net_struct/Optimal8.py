import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

__all__=['Optimal8']
class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
class Optimal8(nn.Module):
    def __init__(self, total_num,first,middle,end, num_classes):
        super(Optimal8, self).__init__()
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
        self.attention3 = nn.Sequential(
            nn.Linear(middle,middle),
            nn.Sigmoid(),
        )
        self.attention4 = nn.Sequential(
            nn.Linear(middle,middle),
            nn.Sigmoid(),
        )
        self.attention5 = nn.Sequential(
            nn.Linear(middle,middle),
            nn.Sigmoid(),
        )
        self.attention6 = nn.Sequential(
            nn.Linear(middle,middle),
            nn.Sigmoid(),
        )
        self.attention7 = nn.Sequential(
            nn.Linear(middle,middle),
            nn.Sigmoid(),
        )
        self.attention8 = nn.Sequential(
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
        attention3 = self.attention3(encoded)
        attention4 = self.attention4(encoded)
        attention5 = self.attention5(encoded)
        attention6 = self.attention6(encoded)
        attention7 = self.attention7(encoded)
        attention8 = self.attention8(encoded)
        # print("attention",attention1.shape)
        f1 = torch.mul(encoded, attention1)
        f2 = torch.mul(encoded, attention2)
        f3 = torch.mul(encoded, attention3)
        f4 = torch.mul(encoded, attention4)
        f5 = torch.mul(encoded, attention5)
        f6 = torch.mul(encoded, attention6)
        f7 = torch.mul(encoded, attention7)
        f8 = torch.mul(encoded, attention8)
        feature1 = self.Globalfeature(f1)
        #feature1=self.l2norm(feature1)
        feature2 = self.Globalfeature(f2)
        #feature2=self.l2norm(feature2)
        feature3 = self.Globalfeature(f3)
        #feature3=self.l2norm(feature3)
        feature4 = self.Globalfeature(f4)
        #feature4=self.l2norm(feature4)
        feature5 = self.Globalfeature(f5)
        #feature5=self.l2norm(feature5)
        
        feature6 = self.Globalfeature(f6)
        #feature6=self.l2norm(feature6)
        feature7 = self.Globalfeature(f7)
        #feature7=self.l2norm(feature7)
        feature8 = self.Globalfeature(f8)
        #feature8=self.l2norm(feature8)
        feature = torch.cat((feature1, feature2), 1)
        feature = torch.cat((feature, feature3), 1)
        feature = torch.cat((feature, feature4), 1)
        feature = torch.cat((feature, feature5), 1)
        feature = torch.cat((feature, feature6), 1)
        feature = torch.cat((feature, feature7), 1)
        feature = torch.cat((feature, feature8), 1)
        # print("feature_shape",feature.shape)


        return [attention1, attention2, attention3, attention4, attention5, attention6, attention7, attention8], [feature1, feature2, feature3, feature4,feature5, feature6, feature7, feature8], feature


