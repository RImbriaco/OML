import torch
from torch import nn
import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        """
        Generalized mean pooling from "Fine-tuning CNN Image Retrieval with
        No Human Annotation" https://arxiv.org/abs/1711.02512
        :param p: Norm power.
        :param eps: Epsilon
        """
        super(GeM, self).__init__()
        self.p = nn.parameter.Parameter(torch.ones(1).cuda() * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)