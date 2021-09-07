import rasterio
import albumentations as alb
import numpy as np
from src.common.hyper_config import band_list
# Band normalization statistics imported from  the original repository
# https://gitlab.tu-berlin.de/rsim/bigearthnet-models-tf/blob/master/BigEarthNet.py
BAND_STATS = {
            'mean': {
                'B01': 340.76769064,
                'B02': 429.9430203,
                'B03': 614.21682446,
                'B04': 590.23569706,
                'B05': 950.68368468,
                'B06': 1792.46290469,
                'B07': 2075.46795189,
                'B08': 2218.94553375,
                'B8A': 2266.46036911,
                'B09': 2246.0605464,
                'B11': 1594.42694882,
                'B12': 1009.32729131
            },
            'std': {
                'B01': 554.81258967,
                'B02': 572.41639287,
                'B03': 582.87945694,
                'B04': 675.88746967,
                'B05': 729.89827633,
                'B06': 1096.01480586,
                'B07': 1273.45393088,
                'B08': 1365.45589904,
                'B8A': 1356.13789355,
                'B09': 1302.3292881,
                'B11': 1079.19066363,
                'B12': 818.86747235
            }
        }


def normalize_resize_ben(band, imsize, band_name):
    mean = BAND_STATS['mean'][band_name]
    std = BAND_STATS['std'][band_name]
    augmenter = alb.Compose([
        alb.Resize(imsize, imsize, always_apply=True),
        alb.Normalize(mean=mean, std=std)
    ])
    return augmenter(image=band)['image']

def norm_band(band):
    """
    Normalize the band between [0,1]
    :param band:
    :return:
    """
    b_min, b_max = band.min(), band.max()
    return (band - b_min) / (b_max - b_min)


def load_rgb(path):
    """
    Load BEN data as 3-band RGB.
    :param path: Path to image.
    :return:
    NumPy array with RGB image.
    """
    bands = band_list['rgb']
    img = None
    fmt = "_{}.tif"
    for b in bands:
        band_ds = rasterio.open(path + fmt.format(b))
        aux = band_ds.read(1)
        aux = norm_band(aux)
        aux = np.expand_dims(aux, axis=-1)
        if img is None:
            img = aux
        else:
            img = np.concatenate((img, aux), axis=-1)
    return img


def load_full(path, imsize=120):
    """
    Load BEN data as multi-spectral images.
    :param path: Path to image.
    :return:
    NumPy array with all spectral bands.
    """
    bands = band_list['full']
    img = None
    fmt = "_{}.tif"
    for b in bands:
        band_ds = rasterio.open(path + fmt.format(b))
        aux = band_ds.read(1)
        aux = normalize_resize_ben(aux, imsize, b)
        aux = np.expand_dims(aux, axis=-1)
        if img is None:
            img = aux
        else:
            img = np.concatenate((img, aux), axis=-1)
    return img
