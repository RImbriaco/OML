from src.retrieval_core.models.encoder import vgg16, vgg19
from src.retrieval_core.models.key_collection import vgg_keys


class VggBuilder:
    def __init__(self, name: str, output_stride: dict, in_dim=3, pretrained=True):
        """
        Builds ResNet variant based on given name and pretrained condition
        :param name: resnet variant
        :param in_dim: in case of 3+ channel images
        :param pretrained: if to use a pretrained (bool)
        """
        super(VggBuilder, self).__init__()

        self._name = name
        self._output_stride = output_stride
        self._pretrained = pretrained
        self._in_dim = in_dim
        self._vgg_map = {
            'vgg16': vgg16,
            'vgg19': vgg19,
        }
        self._vgg_model = self._vgg_map[self._name]

    def _vgg(self):
        if self._name not in vgg_keys:
            raise NotImplementedError
        else:
            model = self._vgg_model(in_channels=self._in_dim,
                                    pretrained=self._pretrained,
                                    )
        return model

    def get_layers(self):
        model = self._vgg()

        return model.features