import os
from typing import Union
from src.retrieval_core.dataloader.augmentations.alb_operations import AugmentOperations
from src.retrieval_core.dataloader.helpers.image_readers import default_loader
import numpy as np


class BaseLoader:
    def __init__(self, mode, root_dir, train_cfg, augment_cfg, transform, seed=2020):
        """
        Base dataloader class.
        :param mode: Train/val/test.
        :param root_dir: Base directory, set in config.
        :param train_cfg: Train configuration.
        :param augment_cfg: Augmentation configuration.
        :param transform: Transform to apply.
        :param seed: Determines train/val/test random split.
        """
        self.mode = mode
        self.root_dir = root_dir
        self.train_cfg = train_cfg
        self.augment_cfg = augment_cfg
        self.transform = transform
        self.seed = seed
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2
        self.img_list, self.labels = self.list_data()
        self.ext_dict = {
            'UCM': '.tif',
            'WHDLD': '.jpg'
        }

    def join(self, paths: Union[list, str]) -> str:
        if isinstance(paths, list):
            return os.path.join(self.root_dir, *paths)
        elif isinstance(paths, str):
            return os.path.join(self.root_dir, paths)
        else:
            raise TypeError('Incorrect type for {}'.format(paths))

    def apply_augmentations(self, img):
        """
        Apply augmentations based on mode.
        :param img: Images to load.
        :return:
        Trasnformed image.
        """
        aug = AugmentOperations(self.augment_cfg)
        if self.mode == 'train':
            img = aug.train_transform(img, False)
        elif self.mode == 'test' or self.mode == 'val':
            img = aug.test_transform(img, False)
        else:
            raise NotImplementedError(
                'Incorrect mode for {}'.format(self.mode)
            )
        return img

    def list_data(self):
        """
        Process csv and generate image and label lists.
        :return:
        Image and label lists.
        """
        csv_path = self.join(['{}.csv'.format(self.mode)])
        with open(csv_path) as f:
            text = f.readlines()
            img_list = [t.split(',') for t in text]
        img_list = img_list
        img_list = np.array(img_list)
        img_names = img_list[:, 0]
        img_labels = img_list[:, 1:].astype('int')
        return img_names, img_labels

    @staticmethod
    def remove_indices(all_idxs, ratio, img_list):
        """
        Remove already selected indices from the image list. Used for
        generating splits.
        :param all_idxs: List of indices.
        :param ratio: Desired split ratio.
        :param img_list: Image list.
        :return:
        Lists of selected and all remaining indices.
        """
        sub_idxs = np.random.choice(all_idxs, int(ratio * len(img_list)), replace=False)
        [all_idxs.remove(i) for i in sub_idxs]
        return sub_idxs, all_idxs

    def split_indices(self):
        """
        Prepare data splits.
        :return:
        Dictionary with data splits.
        """
        img_list, label_list = self.list_data()
        all_idxs = list(range(len(img_list)))
        np.random.seed(self.seed)
        train_idxs, all_idxs = self.remove_indices(
            all_idxs=all_idxs,
            ratio=self.train_ratio,
            img_list=img_list
        )
        val_idxs, all_idxs = self.remove_indices(
            all_idxs=all_idxs,
            ratio=self.val_ratio,
            img_list=img_list
        )
        test_idxs = np.array(all_idxs)
        ret_dict = {
            'train': np.sort(train_idxs),
            'val': np.sort(val_idxs),
            'test': np.sort(test_idxs)
        }
        return img_list[ret_dict[self.mode]], label_list[ret_dict[self.mode]]

    def load_image(self, image_list, item):
        """
        Load images from the list.
        :param image_list: List of images.
        :param item: Torch dataloader item number.
        :return:
        Transformed image.
        """
        ext = self.ext_dict[os.path.split(self.root_dir)[-1]]
        img_path = self.join(['Images', image_list[item] + ext])
        img = default_loader(self.join(img_path))
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
    import yaml
    import matplotlib.pyplot as plt


    with open('/home/rimbriaco/PycharmProjects/rsir20/configs/retrieval_config.yaml', 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    uc = BaseLoader('test',
                   '/home/rimbriaco/PycharmProjects/DATA/RS/WHDLD',
                   config['train'],
                   config['augment_config'],
                   None)

    for i in uc:
        plt.imshow(i['bands10'][0])
        plt.show()