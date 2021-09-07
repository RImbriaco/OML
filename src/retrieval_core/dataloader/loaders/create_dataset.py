import os
from torch.utils import data
from src.retrieval_core.dataloader.loaders.lmdb_loader import BigEarthLMDB
from src.retrieval_core.dataloader.loaders.base_loader import BaseLoader
from src.retrieval_core.dataloader.loaders.aid_loader import AIDLoader
from src.retrieval_core.dataloader.augmentations.alb_operations import AugmentOperations


class CreateDataset:
    def __init__(self, config, dataset, mode):
        """
        Factory class for datasets.
        :param config: Configuration dictionary.
        :param dataset: Dataset name.
        :param mode: Train/val/test.
        """
        super(CreateDataset, self).__init__()
        self._config = config
        self._train_cfg = config['train']
        self._augment_cfg = config['augment_config']
        self._dataset = dataset
        self._mode = mode

        self._bs = self._train_cfg['batch_size']
        self._imsize = self._augment_cfg['image_size']
        self._is_aug = True if self._mode == 'train' else False
        self._root = config['root_dir']
        self.lmdb_path = os.path.join(self._root, 'Splits', 'lmdb')
        self.train_csv = os.path.join(self._root, 'Splits', 'train.csv')
        self.test_csv = os.path.join(self._root, 'Splits', 'test.csv')
        self.val_csv = os.path.join(self._root, 'Splits', 'val.csv')

    def create_dataset(self, mode):
        """
        Instance dataset class.
        :param mode: Train/val/test.
        :return:
        Dataset object instance.
        """
        shuffle = True if mode == 'train' else False
        drop_last = True if mode == 'train' else False
        augmenter = AugmentOperations(self._augment_cfg)

        if self._augment_cfg['use_transform'] and self._mode == 'train':
            transform = augmenter.train_transform
        else:
            transform = augmenter.test_transform

        if self._dataset == 'BigEarthNet':
            dset_gen = BigEarthLMDB(self.lmdb_path, transform, mode, True,
                                    self.train_csv, self.test_csv, self.val_csv)
        elif self._dataset == 'AID':
            dset_gen = AIDLoader(self._mode, self._root, self._train_cfg,
                                 self._augment_cfg, transform)
        else:
            dset_gen = BaseLoader(self._mode, self._root, self._train_cfg,
                                  self._augment_cfg, transform)
        dataloader = data.DataLoader(dset_gen, batch_size=self._bs,
                       num_workers=4, shuffle=shuffle, pin_memory=True, drop_last=drop_last)
        return dataloader

    def deploy_data_loader(self, mode):
        """
        Wrap dataset in dataloader class.
        :param mode: Train/val/test.
        :return:
        Dataloader object.
        """
        data_loader = self.create_dataset(mode)
        return data_loader
