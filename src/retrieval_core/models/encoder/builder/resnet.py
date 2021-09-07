import torch.nn as nn
from src.retrieval_core.models.encoder import resnet18, resnet34
from src.retrieval_core.models.encoder import resnet50, resnet101, resnet152
from src.retrieval_core.models.key_collection import resnet_keys


class ResNetBuilder:
    def __init__(self, name: str, output_stride: dict, in_dim=3, pretrained=True):
        """
        Builds ResNet variant based on given name and pretrained condition
        :param name: resnet variant
        :param in_dim: in case of 3+ channel images
        :param pretrained: if to use a pretrained (bool)
        """
        super(ResNetBuilder, self).__init__()

        self._name = name
        self._output_stride = output_stride
        self._pretrained = pretrained
        self._in_dim = in_dim
        self._resnet_map = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152
        }
        self._resnet_model = self._resnet_map[self._name]

    def _resnet(self):
        if self._name not in resnet_keys:
            raise NotImplementedError
        else:
            model = self._resnet_model(in_channels=self._in_dim,
                                       pretrained=self._pretrained,
                                       stride=self._output_stride['stride'],
                                       dilation=self._output_stride['dilation'])
        return model

    def get_layers(self):
        model = self._resnet()
        # Manually set for layer 0
        layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)

        return [layer0, model.layer1, model.layer2, model.layer3, model.layer4]