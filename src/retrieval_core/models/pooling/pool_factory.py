from torch import nn
from src.retrieval_core.models.pooling.GeM import GeM


class PoolFactory(nn.Module):
    def __init__(self, pool='max'):
        super(PoolFactory, self).__init__()

        pool_type = {
            'avg': nn.AdaptiveAvgPool2d(1),
            'max': nn.AdaptiveMaxPool2d(1),
            'gem': GeM()
        }

        if pool not in pool_type.keys():
            raise ValueError('Unknown pooling methods for {}'.format(pool))
        else:
            self.pool = pool_type[pool]

    def forward(self, x):
        return self.pool(x)



