import torch
import numpy as np
from torch import nn


class BNNeck(nn.Module):
    def __init__(self, in_dim, inter_dim, num_classes):
        """
        Batch-norm neck as defined in: "Bag of Tricks and A Strong Baseline for
        Deep Person Re-identification" https://arxiv.org/abs/1903.07071
        :param in_dim: Input dimension.
        :param inter_dim: Embedding dimension.
        :param num_classes: Number of classes.
        """
        super(BNNeck, self).__init__()
        self.fc1 = nn.Linear(in_dim, inter_dim)
        self.bn = nn.BatchNorm1d(inter_dim)
        self.fc2 = nn.Linear(inter_dim, num_classes, bias=False)

    def forward(self, x):
        fc1 = self.fc1(x)
        x = self.bn(fc1)
        x = self.fc2(x)
        return fc1, x

    def get_dim(self):
        return self.fc1.out_features


class PartBased(nn.Module):
    def __init__(self, splits, pool, in_dim, inter_dim, num_classes):
        """
        Allows for partitioning the input tensor as in: "Learning Discriminative
        Features with Multiple Granularities for Person Re-Identification"
        https://arxiv.org/abs/1804.01438
        :param splits: Part-based splits.
        :param pool: Pooling function.
        :param in_dim: Input dimension.
        :param inter_dim: Embedding dimension.
        :param num_classes: Number of classes.
        """
        super(PartBased, self).__init__()

        self.splits = splits
        self.pool = pool
        self.bnneck_pack = nn.ModuleList([])
        self.bnn_triplet_pack = nn.ModuleList([])

        for s in self.splits:
            self.bnn_triplet_pack.append(BNNeck(in_dim, inter_dim, num_classes))
            for _ in range(s[0] * s[1] * s[2]):
                self.bnneck_pack.append(BNNeck(in_dim // s[0], inter_dim, num_classes))

    def create_partitions(self, x) -> list:
        """
        Prepare partitions according the split list.
        :param x: Input tensor.
        :return:
        List of chunks from the original tensor.
        """
        chunks_list = list([])
        for split in self.splits:
            c_chunks = torch.chunk(x, split[0], dim=1)
            for c_chunk in c_chunks:
                h_chunks = torch.chunk(c_chunk, split[1], dim=2)
                for h_chunk in h_chunks:
                    w_chunks = torch.chunk(h_chunk, split[2], dim=3)
                    for w_chunk in w_chunks:
                        chunks_list.append(self.pool(w_chunk).squeeze(3).squeeze(2))
        return chunks_list

    def forward(self, x):
        logits_pack = ([])
        tensor_chunk_dict = self.create_partitions(x)
        for t, bnn_layer in zip(tensor_chunk_dict, self.bnneck_pack):
            logits_pack.append(bnn_layer(t))

        triplet_pack = ([])
        x = self.pool(x).squeeze(3).squeeze(2)
        for bnn_triplet in self.bnn_triplet_pack:
            triplet_pack.append(bnn_triplet(x))

        return logits_pack, triplet_pack

    def get_embedding_dim(self):
        """
        Compute total embedding dimension based on the number of splits.
        :return:
        Final embedding dimension.
        """
        base_dim = self.bnneck_pack[0].get_dim()
        branch_count = len(self.splits)
        split_count = np.sum([np.prod(sp) for sp in self.splits])
        return (branch_count + split_count) * base_dim

