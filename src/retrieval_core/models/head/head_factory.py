from torch import nn
from .part_based import PartBased
from .single import Single
from src.retrieval_core.models.pooling.pool_factory import PoolFactory


class HeadFactory(nn.Module):
    def __init__(self, parts, pool, in_dim, inter_dim, num_classes):
        """
        Creates instances of the Head module in the CNN.
        :param parts: Number of parts the input is divided into. Each input
        is assigned one Head object.
        :param pool: Pooling function.
        :param in_dim: Input dimensions.
        :param inter_dim: Embedding dimensions.
        :param num_classes: Number of classes.
        """
        super(HeadFactory, self).__init__()
        self.pool = PoolFactory(pool)
        self.parts = parts
        if self.is_single_branch():
            self.head = Single(self.pool, in_dim, num_classes)
        else:
            self.head = PartBased(parts, self.pool, in_dim, inter_dim, num_classes)

    def is_single_branch(self):
        return len(self.parts) == 1 and self.parts[0] == [1, 1, 1]

    def forward(self, x):
        return self.head(x)

