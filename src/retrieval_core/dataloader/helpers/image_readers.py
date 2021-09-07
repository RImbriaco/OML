import cv2
from PIL import Image


def cv_loader(path):
    """
    Load image using OpenCV. Default loader.
    :param path: Path to image.
    :return:
    OpenCV image object.
    """
    return cv2.resize(cv2.imread(path), (256, 256), interpolation=cv2.INTER_AREA)


def pil_loader(path):
    """
    Load image using Pillow. Using this requires extensive modification of
    dataloading scripts. Not recommended.
    :param path: Path to image.
    :return:
    Pillow image object.
    """
    return Image.open(path)


def default_loader(path):
    return cv_loader(path)






