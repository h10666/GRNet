import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HSIC(nn.Module):

    def __init__(self):
        super(HSIC, self).__init__()
        self.function = nn.PairwiseDistance(p=2, eps=1e-06).to(device)

    def forward(self, attention1, attention2):
        bsz = attention1.shape[0]
        m1 = torch.FloatTensor(attention1.shape[0], attention1.shape[1]).zero_()
        m1 = m1.to(device)

        m2 = torch.FloatTensor(attention2.shape[0], attention2.shape[1]).zero_()
        m2 = m2.to(device)

        m1 = self.function(attention1, m1)
        m2 = self.function(attention2, m2)

        m1 = m1.unsqueeze(1)
        m2 = m2.unsqueeze(1)
        # m1.expand(attention1.shape[0],)

        attention1 =attention1.div(m1)  # torch.div() 逐元素除法
        attention2 = attention2.div(m2)

        H = torch.ones(attention1.shape[0], attention1.shape[0]) * (1 / attention1.shape[0]) * (-1) + torch.eye(
            attention1.shape[0])
        H = H.to(device)

        k1 = attention1.mm(attention1.t())
        k2 = attention2.mm(attention2.t())
        hsic = H.mm(k1)
        hsic = hsic.mm(H)
        hsic = hsic.mm(k2)
        # print(hsic)
        hsic = torch.trace(hsic)/bsz
        # print(hsic)
        return hsic