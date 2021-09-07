import random
import numpy as np
from scipy.ndimage import rotate


class ImageTransforms:
    def __init__(self, ops_to_apply=('h_flip', 'v_flip', 'rotate'), p=0.5):
        """
        Only applies one operation at a time with probability of p.
        :param ops_to_apply: Operations to apply
        :param p: probability of applying operation.
        """
        super().__init__()

        self._ops_to_apply = ops_to_apply
        self._p = p
        self._rotate_list = [self._rotate90,
                             self._rotate180,
                             self._rotate270]
        self._ops_map = {
            'h_flip': self._h_flip,
            'v_flip': self._v_flip,
            'rotate': random.choice(self._rotate_list)
        }

    @staticmethod
    def _h_flip(image):
        return np.array([np.fliplr(i) for i in image])

    @staticmethod
    def _v_flip(image):
        return np.array([np.flipud(i) for i in image])

    @staticmethod
    def _rotate90(image):
        return rotate(image, 90, axes=[1,2])

    @staticmethod
    def _rotate180(image):
        return rotate(image, 180, axes=[1,2])

    @staticmethod
    def _rotate270(image):
        return rotate(image, 270, axes=[1,2])


    @staticmethod
    def _convert_to_np(image):
        return np.array(image)

    def _apply_multiple_ops(self):
        """
        Collects operations to apply on provided list of images.
        :return:
        Function list to apply.
        """
        funcs_to_apply = list([])
        for op in self._ops_to_apply:
            if op not in self._ops_map:
                raise ValueError('Operation not implemented or incorrect!')
            else:
                funcs_to_apply.append(self._ops_map[op])
        return funcs_to_apply

    def _get_op_funcs_to_apply(self):
        """
        Gets the functions to applying transformations and picks one
        :return:
        Selected transformation.
        """
        if isinstance(self._ops_to_apply, str):
            if self._ops_to_apply not in self._ops_map:
                raise ValueError('Operation not implemented or incorrect!')
            return self._ops_map[self._ops_to_apply]
        elif isinstance(self._ops_to_apply, tuple):
            return random.choice(self._apply_multiple_ops())
        else:
            raise ValueError('Incorrect type of operation to apply!')

    def _unpack_and_apply_transforms(self, image_pack):
        """
        Transform collection of images.
        :param image_pack: Collection of images.
        :return:
        Transformed images.
        """
        transform_image_pack = dict({})
        func_to_use = self._get_op_funcs_to_apply()

        for (image_key, image) in image_pack.items():
            if image is None:
                transform_image_pack[image_key] = None
            else:
                image = self._convert_to_np(func_to_use(image))
                transform_image_pack[image_key] = image
        return transform_image_pack

    def apply(self, **image_pack):
        """
        :param image_pack: Tuple of images
        :return: applies image transformation and returns image.
        """
        if random.random() < self._p:
            return self._unpack_and_apply_transforms(image_pack)
        else:
            return image_pack

# Transformation definitions.
class HorizontalFlip(ImageTransforms):
    def __init__(self, p):
        super().__init__()
        self._ops_to_apply = 'h_flip'
        self._p = p


class VerticalFlip(ImageTransforms):
    def __init__(self, p):
        super().__init__()
        self._ops_to_apply = 'v_flip'
        self._p = p


class RandomRotation(ImageTransforms):
    def __init__(self, p):
        super().__init__()
        self._ops_to_apply = 'rotate'
        self._p = p


class OneOfImageTransforms:
    def __init__(self, *transforms):
        """
        Picks one of the provided transforms and applies it.
        Currently supports Horizontal, Vertical and Random Rotations by angles of 90.
        :param transforms: list of transforms to apply.
        """
        super().__init__()
        self._transforms = transforms

    def apply(self, **image_pack):
        transform = random.choice(self._transforms)
        return transform.apply(**image_pack)

