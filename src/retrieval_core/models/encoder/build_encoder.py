from torch import nn
from src.retrieval_core.models.key_collection import resnet_keys, resnet_ibn_keys, se_keys, efficientnet_keys, vgg_keys
from src.retrieval_core.models.encoder.builder import VggBuilder
from src.retrieval_core.models.encoder.builder import ResNetBuilder
from src.retrieval_core.models.encoder.builder import ResNetIBNBuilder
from src.retrieval_core.models.encoder.builder import SENetBuilder
from .backbones.resnet import BasicBlock as resnet_basic
from .backbones.resnet import Bottleneck as resnet_bottle
from .backbones.resnet_ibn_a import BasicBlock as resnet_ibn_basic
from .backbones.resnet_ibn_a import Bottleneck as resnet_ibn_bottle
from src.retrieval_core.models.encoder.builder import EfficientNetBuilder


class BuildEncoder(nn.Module):
    def __init__(self,  encoder: str, in_channels: int,
                        pretrained: bool, output_stride: int):
        """
        The forward of the encoder, gets the encoder graph from Encoder class.
        :param encoder: name of the encoder
        :param in_channels: input channels
        :param pretrained: if pretrained
        :param output_stride: output stride to determine the level of downsample
        """
        super().__init__()

        self.encoder_name = encoder
        self.in_channels = in_channels
        build_encoder = Encoder(encoder, in_channels, pretrained, output_stride)
        self.encoder = build_encoder.build()

        if self.encoder_name not in vgg_keys:
            self.encoder_block0 = self.encoder[0]
            self.encoder_block1 = self.encoder[1]
            self.encoder_block2 = self.encoder[2]
            self.encoder_block3 = self.encoder[3]
            self.encoder_block4 = self.encoder[4]

    def _get_vgg_channels(self) -> dict:
        idx = list(self.encoder._modules.keys())[-3]
        return {4: self.encoder._modules[idx].out_channels}

    def _get_resnet_channels(self) -> dict:
        resnet_channel_pack = {0: self.encoder[0][0].out_channels}
        for i in range(1, 5):
            block = self.encoder[i][-1]
            if isinstance(block, resnet_basic):
                resnet_channel_pack[i] = block.conv2.out_channels
            elif isinstance(block, resnet_bottle):
                resnet_channel_pack[i] = block.conv3.out_channels
            else:
                raise RuntimeError('Unknown ResNet block: {}'.format(block))
        return resnet_channel_pack

    def _get_resnet_ibn_channels(self) -> dict:
        resnet_ibn_channel_pack = {0: self.encoder[0][0].out_channels}
        for i in range(1, 5):
            block = self.encoder[i][-1]
            if isinstance(block, resnet_ibn_basic):
                resnet_ibn_channel_pack[i] = block.conv2.out_channels
            elif isinstance(block, resnet_ibn_bottle):
                resnet_ibn_channel_pack[i] = block.conv3.out_channels
            else:
                raise RuntimeError('Unknown ResNet block: {}'.format(block))
        return resnet_ibn_channel_pack

    def _get_senet_channels(self) -> dict:
        senet_channel_pack = {0: self.encoder[0].conv1.out_channels}
        for i in range(1, 5):
            channels = self.encoder[i][-1].conv3.out_channels
            senet_channel_pack[i] = channels
        return senet_channel_pack

    def _get_efficientnet_channels(self) -> dict:
        efficient_channel_pack = dict({})
        for i in range(0, 4):
            efficient_channel_pack[i] = self.encoder[i][-1]._bn2.num_features
        efficient_channel_pack[4] = self.encoder[4][-1].num_features
        return efficient_channel_pack

    def get_channels(self) -> dict:
        channel_map = dict({})
        if self.encoder_name in resnet_keys:
            channel_map = self._get_resnet_channels()
        elif self.encoder_name in resnet_ibn_keys:
            channel_map = self._get_resnet_ibn_channels()
        elif self.encoder_name in se_keys:
            channel_map = self._get_senet_channels()
        elif self.encoder_name in efficientnet_keys:
            channel_map = self._get_efficientnet_channels()
        elif self.encoder_name in vgg_keys:
            channel_map = self._get_vgg_channels()
        return channel_map

    def forward(self, x):
        input_shape = x.size()[2], x.size()[3]
        if self.encoder_name in vgg_keys:
            eb = self.encoder(x)
            return [None, None, None, None, eb], input_shape
        else:
            eb0 = self.encoder_block0(x)
            eb1 = self.encoder_block1(eb0)
            eb2 = self.encoder_block2(eb1)
            eb3 = self.encoder_block3(eb2)
            eb4 = self.encoder_block4(eb3)

        return [eb0, eb1, eb2, eb3, eb4], input_shape


class Encoder:
    def __init__(self, encoder_name: str, in_channels: int,
                 pretrained: bool, output_stride: int):
        """
        Generates the encoder graph from the given parameters

        Args:
            encoder_name: name of the encoder
            in_channels: input channels
            pretrained: if pretrained
            output_stride: output stride to determine the rate of downsample
        """
        super().__init__()

        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.pretrained = pretrained
        # Output stride to strides and dilation rates.
        self.output_stride = output_stride
        self.output_stride_map = {
            8:  {'stride': (1, 1), 'dilation': (2, 4)},
            16: {'stride': (1, 2), 'dilation': (1, 2)},
            32: {'stride': (2, 2), 'dilation': (1, 1)}
        }

        if output_stride not in self.output_stride_map.keys():
            raise NotImplementedError('Output stride not supported', output_stride)
        else:
            self.output_stride_settings = self.output_stride_map[output_stride]

    def _build_vgg(self):
        return VggBuilder(self.encoder_name, self.output_stride_settings,
                             self.in_channels, self.pretrained)

    def _build_resnet(self):
        return ResNetBuilder(self.encoder_name, self.output_stride_settings,
                             self.in_channels, self.pretrained)

    def _build_resnet_ibn(self):
        return ResNetIBNBuilder(self.encoder_name, self.output_stride_settings,
                             self.in_channels, self.pretrained)

    def _build_senet(self):
        return SENetBuilder(self.encoder_name, self.output_stride_settings,
                            self.in_channels, self.pretrained)

    def _build_efficient(self):  # Pre-trained always on for efficient net
        return EfficientNetBuilder(self.encoder_name, self.output_stride,
                                   self.in_channels)

    def get_model_from_key(self):
        if self.encoder_name in resnet_keys:
            return self._build_resnet().get_layers()
        elif self.encoder_name in resnet_ibn_keys:
            return self._build_resnet_ibn().get_layers()
        elif self.encoder_name in se_keys:
            return self._build_senet().get_layers()
        elif self.encoder_name in efficientnet_keys:
            return self._build_efficient().get_layers()
        elif self.encoder_name in vgg_keys:
            return self._build_vgg().get_layers()

    def build(self):
        if self.encoder_name not in (
                resnet_keys + resnet_ibn_keys + se_keys + efficientnet_keys + vgg_keys):
            raise NotImplementedError('Encoder not implemented or incorrect!')
        else:
            return self.get_model_from_key()
