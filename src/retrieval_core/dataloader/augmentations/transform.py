from torchvision.transforms import Compose
from src.retrieval_core.dataloader.loaders.lmdb_loader import ToTensor, Normalize

bands_mean = {
    'bands10_mean': [429.9430203, 614.21682446, 590.23569706, 2218.94553375],
    'bands20_mean': [950.68368468, 1792.46290469, 2075.46795189, 2266.46036911,
                     1594.42694882, 1009.32729131],
    'bands60_mean': [340.76769064, 2246.0605464]
}

bands_std = {
    'bands10_std': [572.41639287, 582.87945694, 675.88746967, 1365.45589904],
    'bands20_std': [729.89827633, 1096.01480586, 1273.45393088, 1356.13789355,
                    1079.19066363, 818.86747235],
    'bands60_std': [554.81258967, 1302.3292881]
}

imagenet_mean = {
    'bands10_mean': [0.485, 0.456, 0.406],
    'bands20_mean': [0.485, 0.456, 0.406],
}

imagenet_std = {
    'bands10_std': [0.229, 0.224, 0.225],
    'bands20_std': [0.229, 0.224, 0.225],
}