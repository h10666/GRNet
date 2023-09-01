
import torch

from torch import nn

__all__=['Exclusivity']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
function = nn.PairwiseDistance(p=2, eps=1e-06).to(device)
class Exclusivity(nn.Module):
    def __init__(self):
        super(Exclusivity, self).__init__()
        self.function = nn.PairwiseDistance(p=2, eps=1e-06).to(device)
    def forward(self,attention1,attention2):
        bsz=attention1.shape[0]
        mask = torch.mul(attention1, attention2)
        # print(mask)
        mask = abs(mask)
        # print(mask)
        mask = mask.sum(dim=1)
        mask = mask.sum(dim=0)
        mask=mask/bsz
        # print(mask)
        return mask
