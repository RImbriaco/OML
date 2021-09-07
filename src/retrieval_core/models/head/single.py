import torch.nn as nn


class Single(nn.Module):
    def __init__(self, pool, in_dim, num_classes):
        """
        Simple Head object.
        :param pool: Pooling function.
        :param in_dim: Input dimensions.
        :param num_classes: Number of classes.
        """
        super(Single, self).__init__()
        self.pool = pool
        self.fc = nn.Linear(in_dim, num_classes, bias=False)

    def forward(self, x):
        x = self.pool(x).squeeze(3).squeeze(2)
        return self.fc(x), x
