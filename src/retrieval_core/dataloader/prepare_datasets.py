import os
import numpy as np
from glob import glob
np.random.seed(24)

"""
Used for generating the train, val and test csv files. Unless the seed is 
changed, the files provided in the repo should be used. 
"""

def write_file(path, idxs, names, labels, fmt):
    """
    Save data to CSV format.
    :param path: Output path.
    :param idxs: Indices to write.
    :param names: List of image names.
    :param labels: List of image labels.
    :param fmt: CSV Formatting.
    """
    with open(path, 'w') as f:
        for i in idxs:
            f.write(fmt.format(names[i], *labels[i]))

def read_file(csv_path):
    """
    Open and parse CSV.
    :param csv_path: Path to csv.
    :return:
    List of image names and labels.
    """
    with open(csv_path) as f:
        text = f.readlines()
        img_list = [t.split(',') for t in text]
    img_list = img_list[1:]
    img_list = np.array(img_list)
    img_names = img_list[:, 0]
    img_labels = img_list[:, 1:].astype('int')
    return img_names, img_labels

def read_folder(csv_path, mode):
    """
    Read folders based on pre-existing splits.
    :param csv_path: Path to csv.
    :param mode: Train/val/test.
    :return:
    File names.
    """
    path_train = os.path.join(os.path.split(csv_path)[0], mode, '*.jpg')
    train_names = glob(path_train)
    train_names = [os.path.splitext(os.path.split(tn)[1])[0] for tn in train_names]
    return train_names

def prepare_aid(csv_path):
    """
    AID has a special structure so needs a custom process.
    :param csv_path: Path to csv.
    """
    img_names, img_labels = read_file(csv_path)

    train_names = read_folder(csv_path, 'train')
    train_idxs = np.sort(
        [np.argwhere(img_names == i)[0][0] for i in train_names])

    test_names = read_folder(csv_path, 'test')
    test_idxs = np.sort(
        [np.argwhere(img_names == i)[0][0] for i in test_names])

    fmt = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n"
    write_file(os.path.join(os.path.split(csv_path)[0], 'train.csv'),
               train_idxs,
               img_names,
               img_labels,
               fmt)

    write_file(os.path.join(os.path.split(csv_path)[0], 'test.csv'),
               test_idxs,
               img_names,
               img_labels,
               fmt)

def prepare_std(csv_path, fmt):
    """
    Parse the existing folders and create accompanying CSVs.
    :param csv_path: PAth to CSV.
    :param fmt: CSV Formatting.
    :return:
    """
    img_names, img_labels = read_file(csv_path)

    idxs = np.arange(len(img_names))
    train_idxs = np.sort(np.random.choice(idxs, int(len(idxs)*0.7), replace=False))

    leftover_idxs = np.array(list(set(idxs).symmetric_difference(set(train_idxs))))
    val_idxs = np.sort(np.random.choice(leftover_idxs, int(len(idxs)*0.1), replace=False))

    test_idxs = np.sort(list(set(leftover_idxs).symmetric_difference(set(val_idxs))))
    write_file(os.path.join(os.path.split(csv_path)[0], 'train.csv'),
               train_idxs,
               img_names,
               img_labels,
               fmt)

    write_file(os.path.join(os.path.split(csv_path)[0], 'val.csv'),
               val_idxs,
               img_names,
               img_labels,
               fmt)

    write_file(os.path.join(os.path.split(csv_path)[0], 'test.csv'),
               test_idxs,
               img_names,
               img_labels,
               fmt)



prepare_aid('/media/csebastian/data/datasets/RSIR/AID/multi-labels.csv')
# prepare_std('/home/rimbriaco/PycharmProjects/DATA/RS/UCM/multi-labels.csv',
#            fmt="{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n")
#prepare_std('/home/rimbriaco/PycharmProjects/DATA/RS/WHDLD/multi-labels.csv',
#              fmt="{}, {}, {}, {}, {}, {}, {}\n")