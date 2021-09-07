import h5py
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from src.common.io.yaml_io import yaml_reader
from src.retrieval_core.dataloader.loaders.create_dataset import CreateDataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path_config', type=str, help='Path to the configuration yaml.')
parser.add_argument('--path_out', type=str, help='Full path of the hdf5 file.')
parser.add_argument('--threshold', type=str, help='Triplet of threshold values as string.')
parser.add_argument('--mode', type=str, help='Comma-separated list with modes to pre-process (e.g. train test and/or val).')


def write_gt_file(path_out, targets, mode, threshold=(0.2, 0.4, 0.6)):
    """
    Prepare the GT file for BEN. This computes the positive matches for all
    difficulty thresholds.
    :param path_out: Output file to write the HDF5 file.
    :param targets: One-hot encoded multi-labels.
    :param mode: Train, validation or test. Each BEN split should be processed
    separately.
    :param threshold: Values that determine the difficulty of the retrieval task.
    """

    with h5py.File(path_out, 'a') as root:
        mode_grp = root.require_group(mode)
        for idx, tgt in enumerate(tqdm(targets)):
            grp = mode_grp.require_group(str(idx))

            target_distance = cdist(np.expand_dims(tgt, axis=0), targets, metric='jaccard')
            _, easy_matches = np.nonzero(target_distance <= threshold[0])
            _, medium_matches = np.nonzero(target_distance <= threshold[1])
            _, hard_matches = np.nonzero(target_distance <= threshold[2])
            hard = hard_matches.astype(np.int32)
            medium = medium_matches.astype(np.int32)
            easy = easy_matches.astype(np.int32)
            grp.require_dataset('hard', hard.shape, dtype=hard.dtype,
                                data=hard)
            grp.require_dataset('medium', medium.shape, dtype=medium.dtype,
                                data=medium)
            grp.require_dataset('easy', easy.shape, dtype=easy.dtype,
                                data=easy)
            grp.require_dataset('tgt', tgt.shape,
                                dtype=tgt.dtype, data=tgt)


def parse(str_list, dt):
    return np.array(str_list.split(','), dtype=dt)


if __name__ == '__main__':
    args = parser.parse_args()
    path_config = args.path_config
    path_out = args.path_out
    threshold = parse(args.threshold, float)
    mode = parse(args.mode, str)

    config = yaml_reader(path_config)
    for m in mode:
        data_loader = CreateDataset(config, 'BigEarthNet', m).create_dataset(m)

        y_true = []
        for i, data in enumerate(tqdm(data_loader)):
            target = data["label"]
            y_true += list(target.cpu().numpy().astype('int'))
        y_true = np.asarray(y_true)
        write_gt_file(path_out, y_true, m, threshold=threshold)


