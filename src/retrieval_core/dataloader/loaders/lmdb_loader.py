import os
import csv
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import pyarrow as pa
import lmdb
import torch
from skimage.transform import resize
import torchvision.transforms as tvt
""" Obtained from original repo 
https://gitlab.tubit.tu-berlin.de/rsim/bigearthnet-models-pytorch"""


def interp_band(bands, img10_shape=[120, 120]):
    """
    https://github.com/lanha/DSen2/blob/master/utils/patches.py
    """
    bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)

    for i in range(bands.shape[0]):
        bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000

    return bands_interp


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class BigEarthLMDB:
    def __init__(self, bigEarthPthLMDB=None, imgTransform=None, state='train', upsampling=True,
                 train_csv=None, val_csv=None, test_csv=None):

        self.env = lmdb.open(bigEarthPthLMDB, readonly=True, lock=False, readahead=False, meminit=False)
        self.imgTransform = imgTransform
        self.train_bigEarth_csv = train_csv
        self.val_bigEarth_csv = val_csv
        self.test_bigEarth_csv = test_csv
        self.state = state
        self.upsampling = upsampling
        self.patch_names = []
        self.readingCSV()

    def readingCSV(self):
        if self.state == 'train':
            with open(self.train_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    self.patch_names.append(row[0])

        elif self.state == 'val':
            with open(self.val_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    self.patch_names.append(row[0])
        else:
            with open(self.test_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    self.patch_names.append(row[0])

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.get(idx[0]), self.get(idx[1]), self.get(idx[2])
        else:
            return self.get(idx)

    def get(self, idx):
        patch_name = self.patch_names[idx]

        if not self.upsampling:
            return self._getData(patch_name, idx)
        else:
            return self._getDataUp(patch_name, idx)

    def _getData(self, patch_name, idx):

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode())

        bands10, bands20, _, multiHots = loads_pyarrow(byteflow)

        sample = {'bands10': bands10.astype(np.float32),
                  'bands20': bands20.astype(np.float32),
                  'label': multiHots.astype(np.float32),
                  'patch_name': patch_name,
                  'idx': idx}

        if self.imgTransform is not None:
            sample = self.imgTransform(sample)

        return sample

    def _getDataUp(self, patch_name, idx):

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode())

        bands10, bands20, _, multiHots = loads_pyarrow(byteflow)

        bands20 = interp_band(bands20)

        sample = {'bands10': bands10.astype(np.float32),
                  'bands20': bands20.astype(np.float32),
                  'label': multiHots.astype(np.float32),
                  'patch_name': patch_name,
                  'idx': idx}

        if self.imgTransform is not None:
            sample = self.imgTransform(sample)

        return sample


class Normalize(object):
    def __init__(self, bands_mean, bands_std):

        self.bands10_mean = bands_mean['bands10_mean']
        self.bands10_std = bands_std['bands10_std']

        self.bands20_mean = bands_mean['bands20_mean']
        self.bands20_std = bands_std['bands20_std']

    def __call__(self, sample):

        band10, band20, label, name, idx = sample['bands10'], sample['bands20'], \
                                           sample['label'], sample[
                                               'name'], sample['idx']

        for t, m, s in zip(band10, self.bands10_mean, self.bands10_std):
            t.sub_(m).div_(s)

        for t, m, s in zip(band20, self.bands20_mean, self.bands20_std):
            t.sub_(m).div_(s)

        sample = {'bands10': band10,
                  'bands20': band20,
                  'label': label, 'name': name, 'idx': idx}
        return sample


class NormalizeImagenet(object):
    def __init__(self, bands_mean, bands_std):

        self.bands10_mean = bands_mean['bands10_mean']
        self.bands10_std = bands_std['bands10_std']

        self.bands20_mean = bands_mean['bands20_mean']
        self.bands20_std = bands_std['bands20_std']

    def __call__(self, sample):

        band10, band20, label, name, idx = sample['bands10'], sample['bands20'], \
                                           sample['label'], sample[
                                               'name'], sample['idx']

        band10 = band10[:3]

        for t, m, s in zip(band10, self.bands10_mean, self.bands10_std):
            t.sub_(m).div_(s)


        sample = {'bands10': band10,
                  'bands20': band10,
                  'label': label, 'name': name, 'idx': idx}
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        band10, band20, label, name, idx = sample['bands10'], sample['bands20'], sample['label'], sample['patch_name'], sample['idx']

        sample = {'bands10': torch.tensor(band10), 'bands20': torch.tensor(band20),
                  'label': label, 'name': name, 'idx':idx}
        return sample





