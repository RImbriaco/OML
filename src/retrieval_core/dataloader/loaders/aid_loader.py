import os
import numpy as np
from src.retrieval_core.dataloader.helpers.image_readers import default_loader
from src.retrieval_core.dataloader.loaders.base_loader import BaseLoader


class AIDLoader(BaseLoader):
    def __init__(self, mode, root_dir, train_cfg, augment_cfg, transform, seed=2020):
        """
        Specialized dataloader for AID dataset.
        :param mode: Train/val/test.
        :param root_dir: Base directory, set in config.
        :param train_cfg: Train configuration.
        :param augment_cfg: Augmentation configuration.
        :param transform: Transform to apply.
        :param seed: Determines train/val/test random split.
        """
        super(AIDLoader, self).__init__(
            mode=mode,
            root_dir=root_dir,
            train_cfg=train_cfg,
            augment_cfg=augment_cfg,
            transform=transform,
            seed=seed
        )
        self.img_list, self.labels = self.list_data()

    # Unconventional naming due to inheritance
    def split_indices(self):
        """
        Process the images and labels based on dataset structure.
        :return:
        Lists containing images and labels for the dataset.
        """
        image_list = list([])
        label_list = list([])
        image_name_list, labels = self.list_data()
        for p in os.listdir(self.join(self.mode)):
            filename, ext = os.path.splitext(p)
            idx = np.argwhere(image_name_list == filename)
            image_list.append(filename)
            # Double indexing due to internal structuring of np array
            label_list.append(labels[idx][0][0])
        return image_list, label_list


    def load_image(self, image_list, item):
        """
        Load images from the list.
        :param image_list: List of images.
        :param item: Torch dataloader item number.
        :return:
        Transformed image.
        """
        img = default_loader(self.join([self.mode, image_list[item] + '.jpg']))
        img = img.transpose((2, 0, 1)).astype(np.float32)
        label = self.labels[item]
        sample = {
            'bands10': img / 255.0,
            'bands20': img / 255.0,
            'label': label.astype(np.float32),
            'patch_name': ' ',
            'idx': item
        }
        sample = self.apply_augmentations(sample)
        return sample

    def __getitem__(self, item):
        image_dict = self.load_image(self.img_list, item)
        return image_dict

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    import shutil
    import os
    base_dir = '/media/csebastian/data/datasets/RSIR/AID/images_tr'
    target_dir = '/media/csebastian/data/datasets/RSIR/AID/train'
    for classes in os.listdir(base_dir):
        for f in os.listdir(os.path.join(base_dir, classes)):
            shutil.copy(os.path.join(base_dir, classes, f), target_dir)