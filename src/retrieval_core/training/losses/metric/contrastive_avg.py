import torch
import numpy as np
import torch.nn as nn
from scipy.spatial.distance import cdist


class ContrastiveAverage(nn.Module):
    def __init__(self, margin=1.2, threshold=0.5):
        super(ContrastiveAverage, self).__init__()
        self.margin = margin
        self.threshold = threshold
        if self.margin is None:  # use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def loss(self, anchor, pos, neg):
        anchor = torch.nn.functional.normalize(anchor, dim=-1)
        pos = torch.nn.functional.normalize(pos, dim=-1)
        neg = torch.nn.functional.normalize(neg, dim=-1)
        pos_distance = torch.norm(anchor - pos, dim=1)
        neg_distance = torch.norm(anchor - neg, dim=1)
        loss = 0.5 * torch.pow(pos_distance, 2) + \
               0.5 * torch.pow(torch.clamp(self.margin - neg_distance, min=0), 2)

        return loss.sum()

    def forward(self, embedding, labels):
        label_distances = cdist(labels.data.cpu(), labels.data.cpu(), 'jaccard')
        label_positives = label_distances <= self.threshold
        label_positives = label_positives - np.eye(label_positives.shape[0])
        anchor = []
        pos = []
        neg = []
        for i in range(label_positives.shape[0]):
            if label_positives[i, :].sum() > 0:
                positives = embedding[np.nonzero(label_positives[i, :])]
                negatives = embedding[np.nonzero(1 - label_positives[i, :])]
                anchor.append(embedding[i])
                pos.append(positives.mean(dim=0))
                neg.append(negatives.mean(dim=0))

        return self.loss(torch.stack(anchor), torch.stack(pos), torch.stack(neg))

