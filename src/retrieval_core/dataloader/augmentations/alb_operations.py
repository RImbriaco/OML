import numpy as np
import albumentations as alb
import src.retrieval_core.dataloader.augmentations.image_transforms as im_t
from .random_erasing import RandomErasing
from torchvision.transforms import Compose
from src.retrieval_core.dataloader.loaders.lmdb_loader import (
    ToTensor,
    Normalize,
    NormalizeImagenet
)
from src.retrieval_core.dataloader.augmentations.transform import (
    bands_mean,
    bands_std,
    imagenet_mean,
    imagenet_std
)


class AugmentOperations:
    def __init__(self, config):
        """
        Applies augmentations from configuration yaml config_file
        Image level augmentation such as flips, rotations are applied only all
        input images.
        Pixel level augmentations are applied only to RGB images.

        Args:
            rgb: rgb image (any 3 channel image)
        """

        super().__init__()
        self._nc = self.none_check
        self.config = config

        self._blur = config['blur']
        self._bright = config['brightness']
        self._contrast = config['contrast']
        self._gamma = config['gamma']
        self._h_flip = config['h_flip']
        self._v_flip = config['v_flip']
        self._rotate = config['rotate']
        self._compose = config['compose']
        self._norm = config['normalize']
        self._imsize = config['image_size']
        self._rand_erase = config['random_erasing']

    @staticmethod
    def none_check(image=None, func=None):
        """
        If image and function is given, applies the function on the image.
        If image is None, returns None.
        If image is passed without a function, returns image

        Unexpected behaviour:
        If function is passed without image, returns None

        Args:
            image:
            func: function to apply on image

        """
        def pass_(empty_image):
            return empty_image

        if func is None:
            func = pass_
        if image is not None:
            image = func(image)
        return image

    def _pixel_ops(self):
        """
        Predefined pixel level operations
        """
        list_ops = [
                    alb.Blur(blur_limit=3, p=self._blur),
                    alb.RandomBrightnessContrast(brightness_limit=0.1, p=self._bright),
                    alb.RandomBrightnessContrast(contrast_limit=0.1, p=self._contrast),
                    alb.RandomGamma(gamma_limit=(90, 100), p=self._gamma)]
        return list_ops

    def _add_composition(self, list_ops):
        """
        Compose operations with append to the original operations
        :param list_ops: list of operations
        :return:
        Operations to apply
        """
        if self._compose:
            ops_to_pass = list_ops + [alb.Compose(list_ops)]
        else:
            ops_to_pass = list_ops
        return ops_to_pass

    def _one_of_pixel_ops(self):
        return alb.OneOf(self._add_composition(self._pixel_ops()), p=0.5)

    def _image_augmentation(self, image):
        """
        Augment image.
        :param image: Input image.
        :return:
        Transformed image.
        """
        im_transform = im_t.OneOfImageTransforms(
            im_t.HorizontalFlip(p=self._h_flip),
            im_t.VerticalFlip(p=self._v_flip),
            im_t.RandomRotation(p=self._rotate)
        )
        # Concatenate to ensure identical transform on all bands
        img = np.vstack((image['bands10'], image['bands20']))
        img = im_transform.apply(rgb=img)
        image['bands10'] = img['rgb'][:4, :]
        image['bands20'] = img['rgb'][4:, :]
        return image

    def _apply_pixel_augmentations(self, image):
        """
        Applies pixel-level operations on the rgb image.
        :param image: Input image.
        :return:
        Transformed image.
        """
        pixel_ops = self._one_of_pixel_ops()
        img = np.vstack((image['bands10'], image['bands20']))
        pixel_aug = pixel_ops(image=img)
        img = pixel_aug['image']
        return img

    def _apply_normalization(self, rgb_image):
        """
        Normalize images by standard setting of mean and variance.
        :param rgb_image: 3-band image.
        :return:
        Normalized 3-band image.
        """
        if self._norm:
            rgb_image = self._nc(rgb_image, self.__normalize_rgb)
        return rgb_image

    def _normalize_inputs(self, image, ben):
        """
        Normalize and stack inputs. To be used during test time.
        :param image: Input image.
        :param ben: Bool for whether BEN is used or not.
        :return:
        Transformed image.
        """
        if ben:
            trans = Compose([ToTensor(),
                             Normalize(bands_mean=bands_mean,
                                       bands_std=bands_std)])
        else:
            trans = Compose([ToTensor(),
                             NormalizeImagenet(bands_mean=imagenet_mean,
                                              bands_std=imagenet_std)])
        return trans(image)

    def _apply_rand_erase(self, image):
        """
        Apply random erasing. NOTE: Unused for publication hence not
        fully tested.
        :param image: Input image.
        :return:
        Transformed image.
        """
        p = self._rand_erase['p']
        sl = self._rand_erase['sl']
        sh = self._rand_erase['sh']
        r1 = self._rand_erase['r1']
        return RandomErasing(p=p, sl=sl, sh=sh, r1=r1).apply(image)

    def _apply_augmentations(self, image, ben):
        """
        Applies the specified augmentation(s).
        :param image: Input image.
        :param ben: Bool for whether BEN is used or not.
        :return:
        Transformed image(s).
        """
        if ben:
            trans = Compose([ToTensor(),
                             Normalize(bands_mean=bands_mean, bands_std=bands_std)])
        else:
            trans = Compose([ToTensor(),
                             NormalizeImagenet(bands_mean=imagenet_mean, bands_std=imagenet_std)])
        img = self._image_augmentation(image)
        img = self._nc(img, self._apply_pixel_augmentations)
        img = self._apply_rand_erase(img)
        image['bands10'] = img[:4, :]
        image['bands20'] = img[4:, :]
        return trans(image)

    def train_transform(self, image, ben=True):
        return self._apply_augmentations(image, ben)

    def test_transform(self, image, ben=True):
        return self._normalize_inputs(image, ben)