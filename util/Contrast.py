
import torch

from torch import nn


__all__=['Contrast']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Contrast(nn.Module):
    def __init__(self):
        super(Contrast, self).__init__()
        self.one=torch.tensor([1.0]).cuda()
        self.function=nn.PairwiseDistance(p=2, eps=1e-06).to(device)
    def forward(self,attention1,attention2):
        bsz=attention1.shape[0]
        distance = self.function(attention1,attention2)
        dis_pow = torch.pow(distance, 2)
        dis_mask = torch.le(dis_pow, 1.0).float()
        dis_loss = torch.sub(self.one, dis_pow)
        dis_loss = torch.mul(dis_loss, dis_mask).sum()
        dis_loss=dis_loss/bsz
        return dis_loss