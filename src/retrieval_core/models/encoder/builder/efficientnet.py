import torch.nn as nn
from src.retrieval_core.models.encoder import EfficientNet


class EfficientNetBuilder:
    def __init__(self, name: str, output_stride: int, in_dim=3):
        """
        Builds EfficientNet variant based on given name

        Args:
            name:  EfficientNet variant
            output_stride: output stride setting to avoid signal decimation
            in_dim: in case of 3+ channel images
        """
        super(EfficientNetBuilder, self).__init__()

        self._name = name
        self._output_stride = output_stride
        self._in_dim = in_dim

        self.block_breaker_map = {
            'efficientnet-b0': (1, 3, 5, 11),
            'efficientnet-b1': (2, 5, 8, 16),
            'efficientnet-b2': (2, 5, 8, 16),
            'efficientnet-b3': (2, 5, 8, 18),
            'efficientnet-b4': (2, 6, 10, 22),
            'efficientnet-b5': (3, 8, 13, 27),
            'efficientnet-b6': (3, 9, 15, 31),
            'efficientnet-b7': (4, 11, 18, 38),
        }

    def get_layers(self):
        model = EfficientNet.from_pretrained(model_name=self._name,
                                             in_channels=self._in_dim,
                                             output_stride=self._output_stride)

        blocks = model._blocks
        bb_pts = self.block_breaker_map[self._name]

        block0_sub = blocks[0: bb_pts[0]]
        block0 = nn.Sequential(model._conv_stem, model._bn0, *block0_sub)
        block1 = nn.Sequential(*blocks[bb_pts[0]: bb_pts[1]])
        block2 = nn.Sequential(*blocks[bb_pts[1]: bb_pts[2]])
        block3 = nn.Sequential(*blocks[bb_pts[2]: bb_pts[3]])
        block4 = nn.Sequential(*blocks[bb_pts[3]: len(blocks)], model._conv_head, model._bn1)
        return [block0, block1, block2, block3, block4]
