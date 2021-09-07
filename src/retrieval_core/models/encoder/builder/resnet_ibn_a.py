import torch.nn as nn
from src.retrieval_core.models.encoder import resnet50_ibn_a, resnet101_ibn_a, resnet152_ibn_a
from src.retrieval_core.models.key_collection import resnet_ibn_keys


class ResNetIBNBuilder:
    def __init__(self, name: str, output_stride: dict, in_dim=3, pretrained=True):
        """
        Builds ResNet variant based on given name and pretrained condition
        :param name: resnet variant
        :param in_dim: in case of 3+ channel images
        :param pretrained: if to use a pretrained (bool)
        """
        super(ResNetIBNBuilder, self).__init__()

        self._name = name
        self._pretrained = pretrained
        self._resnet_map = {
            'resnet50_ibn_a': resnet50_ibn_a,
            'resnet101_ibn_a': resnet101_ibn_a,
            'resnet152_ibn_a': resnet152_ibn_a
        }
        self._resnet_model = self._resnet_map[self._name]

    def _resnet(self):
        if self._name not in resnet_ibn_keys:
            raise NotImplementedError
        else:
            model = self._resnet_model(pretrained=self._pretrained)
        return model

    def get_layers(self):
        model = self._resnet()
        # Manually set for layer 0
        layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)

        return [layer0, model.layer1, model.layer2, model.layer3, model.layer4]