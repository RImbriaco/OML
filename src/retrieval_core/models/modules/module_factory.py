from torch import nn
from .da import DAModule
from .cbam import CBAMModule


class ModuleFactory(nn.Module):
    def __init__(self, module, in_dim=2048):
        super(ModuleFactory, self).__init__()

        module_type = {
            'none': None,
            'da': DAModule(in_dim=in_dim),
            'cbam': CBAMModule(in_dim)
        }

        if module not in module_type.keys():
            raise ValueError('Unknown module for {}'.format(module))
        else:
            self.module = module_type[module]

    def forward(self, x):
        if self.module is None:
            return x
        else:
            return self.module(x)
