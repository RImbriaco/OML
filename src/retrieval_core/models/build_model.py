from torch import nn
from .encoder.build_encoder import BuildEncoder
from .modules.module_factory import ModuleFactory
from .head.head_factory import HeadFactory

class BuildModel(nn.Module):
    def __init__(
            self,
            encoder='resnet101',
            pretrained=True,
            output_stride=8,
            module='da',
            parts=(1,),
            pool='avg',
            inter_dim=256,
            num_classes=2,
            local=False,
            in_channels=3):
        """
        Instance model object based on config.
        :param encoder: Encoder type.
        :param pretrained: Use pre-trained weights or not.
        :param output_stride: Output stride.
        :param module: Attention module.
        :param parts: Parts to split input tensor into.
        :param pool: Pooling function.
        :param inter_dim: Intermediate dimension.
        :param num_classes: Number of classes.
        :param local: Deprecated.
        :param in_channels: Input channels.
        """
        super().__init__()

        self.encoder = BuildEncoder(
            encoder=encoder,
            in_channels=in_channels,
            pretrained=pretrained,
            output_stride=output_stride
        )
        self.layer_dims = self.encoder.get_channels()
        self.module = ModuleFactory(module, self.layer_dims[4])
        self.head = HeadFactory(
            parts=parts,
            pool=pool,
            in_dim=self.layer_dims[4],
            inter_dim=inter_dim,
            num_classes=num_classes
        )
        self.local = local

    def forward(self, x):
        eb_pack, input_shape = self.encoder(x)
        x = self.module(eb_pack[-1])
        return self.head(x)

