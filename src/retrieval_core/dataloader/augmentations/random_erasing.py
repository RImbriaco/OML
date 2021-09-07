import math
import random


class RandomErasing:
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3,
                 mean=(429.9430203, 614.21682446, 590.23569706, 2218.94553375,
                       950.68368468, 1792.46290469, 2075.46795189, 2266.46036911,
                       1594.42694882, 1009.32729131)):
        """
        Applies Random Erasing as per https://arxiv.org/abs/1708.04896
        Mean is defined according to BEN. If working with RE and other datasets
        changing the mean values is necessary.

        :param p: probability of applying operation.
        """
        super().__init__()
        self._p = p
        self._sl = sl
        self._sh = sh
        self._r1 = r1
        self.mean = mean

    def _erase(self, image):
        for attempt in range(100):
            area = image.shape[1] * image.shape[2]

            target_area = random.uniform(self._sl, self._sh) * area
            aspect_ratio = random.uniform(self._r1, 1 / self._r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < image.shape[1] and h < image.shape[2]:
                x1 = random.randint(0, image.shape[1] - h)
                y1 = random.randint(0, image.shape[2] - w)
                # Completely remove the selected patch.
                if image.shape[0] == 3:
                    image[:, x1:x1 + h, y1:y1 + w] = 0
                else:
                    image[:, x1:x1 + h, y1:y1 + w] = 0
        return image

    def apply(self, image):
        if random.random() < self._p:
            return self._erase(image)
        else:
            return image






