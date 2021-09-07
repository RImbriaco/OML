from src.retrieval_core.models.encoder import se_resnet50, se_resnet101, se_resnet152
from src.retrieval_core.models.encoder import se_resnext50_32x4d, se_resnext101_32x4d
from src.retrieval_core.models.encoder import senet154
from src.retrieval_core.models.key_collection import se_keys


class SENetBuilder:
    def __init__(self, name: str, output_stride: dict, in_dim=3, pretrained=True):
        """
        Builds SENet(ResNext) variant based on given name and pretrained condition
        :param name: resnext variant (string)
        :param in_dim: in case of 3+ channel images
        :param pretrained: if to use a pretrained (bool)
        """
        super(SENetBuilder, self).__init__()

        self._name = name
        self._output_stride = output_stride
        self._pretrained = 'imagenet' if pretrained else None
        self._in_dim = in_dim
        self._senet_maps = {
            'se_resnet50': se_resnet50,
            'se_resnet101': se_resnet101,
            'se_resnet152': se_resnet152,
            'se_resnext50_32x4d': se_resnext50_32x4d,
            'se_resnext101_32x4d': se_resnext101_32x4d,
            'senet154': senet154
        }
        self._senet_model = self._senet_maps[self._name]

    def _senet(self):
        if self._name not in se_keys:
            raise NotImplementedError
        else:
            model = self._senet_model(in_channels=self._in_dim,
                                      pretrained=self._pretrained,
                                      strides=self._output_stride['stride'],
                                      dilations=self._output_stride['dilation'])
        return model

    def get_layers(self):
        model = self._senet()
        layer0 = model.layer0
        layer_pack = [model.layer1, model.layer2, model.layer3, model.layer4]

        return [layer0, *layer_pack]
