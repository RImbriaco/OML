# requirements:
#    python 3.x
#    torch = 1.1.0

import torch
from scipy.spatial.distance import jaccard, cdist
import numpy as np
from src.common.evaluation.multilabel_metrics import MultiLabelMetrics

"""
Base function and Smooth-AP implementation obtained from 
https://github.com/Andrew-Brown1/Smooth_AP
"""

def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def compute_aff(x):
    """computes the affinity matrix between an input vector and itself"""
    return torch.mm(x, x.t())


class OML(torch.nn.Module):
    def __init__(self, anneal=0.01):
        """
        Ordered Multi-label Loss. Builds on top of Smooth-AP for differentiable
        multi-label ranking.
        :param anneal: Temperature of the sigmoid.
        """
        super(OML, self).__init__()

        self.anneal = anneal

    @staticmethod
    def remove_diagonal(tensor, batch_size):
        tensor = tensor.flatten()[1:]
        return tensor.view(batch_size - 1, batch_size + 1)[:, :-1].reshape(
            batch_size, batch_size - 1)

    def OrderedMultilabelLoss(self, embeddings, labels):
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        batch_size = labels.shape[0]

        mask = 1.0 - torch.eye(batch_size)
        mask = mask.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        sim_all = compute_aff(embeddings)
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask.cuda()
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1
        ap = torch.zeros(labels.shape[1]).cuda()
        # OML
        for i in range(labels.shape[1]):
            query_idxs = torch.nonzero(labels[:, i] == 1)
            # Avoid the trivial case
            if query_idxs.shape[0] > 1:
                query_idxs = query_idxs.squeeze()
                mask_pos = 1.0 - torch.eye(len(query_idxs))
                mask_pos = mask_pos.unsqueeze(dim=0).repeat(len(query_idxs), 1, 1)
                sim_pos = compute_aff(embeddings[query_idxs])
                sim_pos_repeat = sim_pos.unsqueeze(dim=1).repeat(1, len(query_idxs), 1)
                # compute the difference matrix
                sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 2, 1)
                # pass through the sigmoid
                sim_sg = sigmoid(sim_pos_diff, temp=self.anneal) * mask_pos.cuda()
                sim_pos_rk = torch.sum(sim_sg, dim=-1) + 1

                den = sim_all_rk.index_select(0, query_idxs).index_select(1, query_idxs)
                pos_divide = torch.sum(sim_pos_rk / den)
                ap[i] += (pos_divide / len(query_idxs)) / batch_size

        # Don't use classes with no samples
        ap = ap[torch.nonzero(labels.sum(dim=0) > 1).squeeze()]
        return (1 - ap).mean()

    def forward(self, embeddings, labels):
        return self.OrderedMultilabelLoss(embeddings, labels)
