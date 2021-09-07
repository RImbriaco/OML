import numpy as np
import h5py
from tqdm import tqdm
from multiprocessing import pool
from functools import partial
from scipy.spatial.distance import jaccard, cdist
from src.common.evaluation.multilabel_metrics import MultiLabelMetrics
import torch


# Code for computation of the AP was obtained from the following repository:
# [1] https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/utils/evaluate.py

THRESHOLDS = (0.6, 0.4, 0.2)

def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.

    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images

    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def mp_ap(i, ranks, gnd, kappas=(10,)):
    """
    Adaptation of code from [1] for speedier evaluation with large datasets.
    """
    qgndj = np.array(gnd[i]['junk'])
    qgnd = np.array(gnd[i]['ok'])

    # no positive images, skip from the average
    if qgnd.shape[0] == 0:
        return float('nan'), float('nan'), 1

    # sorted positions of positive and junk images (0 based)
    pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
    junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

    k = 0
    ij = 0
    if len(junk):
        # decrease positions of positives based on the number of
        # junk images appearing before them
        ip = 0
        while ip < len(pos):
            while ij < len(junk) and pos[ip] > junk[ij]:
                k += 1
                ij += 1
            pos[ip] = pos[ip] - k
            ip += 1

    # compute ap
    ap = compute_ap(pos, len(qgnd))
    pos += 1
    prs = np.zeros((len(kappas)))
    for j in np.arange(len(kappas)):
        kq = min(max(pos), kappas[j])
        prs[j] = (pos <= kq).sum() / kq
    return ap, prs, 0


def compute_map(ranks, gnd, kappas=[10]):
    """
    Computes the mAP for a given set of returned results.

         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only

           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query

         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    nq = len(gnd)  # number of queries

    try:
        parfunc = partial(mp_ap, ranks=ranks, gnd=gnd, kappas=kappas)
        with pool.Pool(4) as pp:
            results = pp.map(parfunc, np.arange(nq))
    except OSError:
        results = []
        for i in np.arange(nq):
            results.append(mp_ap(i, ranks, gnd, kappas))
    results = np.array(results)
    map = np.nansum(results[:, 0])
    pr = np.nansum(results[:, 1])
    nempty = results[:, 2].sum()
    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, pr


def compute_map_and_print(ranks, gnd, kappas=[10]):
    gnd_t = []
    for i in gnd:
        g = {}
        g['ok'] = i
        g['junk'] = []
        gnd_t.append(g)
    map, pr = compute_map(ranks, gnd_t, kappas)
    return map, pr


def compute_retrieval_map_and_print(dataset, ranks, gnd, kappas=(1, 5, 10, 20, 50, 100)):
    mean_ap, aps, pr, prs = compute_map(ranks, gnd, kappas)
    print('>> {}: mAP: {:.2f}'.format(dataset, mean_ap * 100))
    print('>> {}: mP@k{} {}'.format(dataset, kappas,
                                    np.round(pr * 100, decimals=2)))

    return mean_ap, aps, pr, prs


def remove_overlap(gt_idx, qr_idx):
    """
    Removes queries from the set of correct matches and makes them
    mutually exclusive.
    :param gt_idx: Indices of ground-truth images.
    :param qr_idx: Indices of query images.
    :return:
    Non-query matching images.
    """
    return gt_idx[np.logical_not(np.in1d(gt_idx, qr_idx))]


def ben_retrieval(retrieval_conf, vectors, targets, mode='test'):
    """
    Processes the special evaluation case with BEN. Due to itsd size the
    positive matches are not computed on-the-fly but pre-computed and stored in
    an HDF5 file (path_gt in config).
    :param retrieval_conf: Retrieval configuration.
    :param vectors: Matrix of image embeddings.
    :param targets: One-hot encoded target mutli-labels.
    :param mode: Train or test.
    :return:
    Dictionary with performance metrics.
    """

    chunk_size = retrieval_conf['chunk_size']
    np.random.seed(retrieval_conf['seed'])
    key_list = ['easy', 'medium', 'hard']
    aux_dict = dict()
    result_dict = dict()
    for k in key_list:
        result_dict[k] = dict()
        for m in ['map', 'prs']:
            result_dict[k][m] = 0.0
        for m in ['ACG@100', 'nDCG@100', 'wAP@100']:
            result_dict[m] = 0.0
    # Open the GT file.
    with h5py.File(retrieval_conf['path_gt'], 'r') as root:
        gt = root[mode]
        # Normalize the vectors
        vectors = np.asarray(vectors)
        vectors = vectors / np.expand_dims(np.linalg.norm(vectors, axis=1), axis=1)
        img_count = len(vectors)
        # Select the queries and database images
        qr_idx = np.sort(np.random.choice(range(img_count), int(img_count*0.2), replace=False))
        db_idx = set(range(img_count)).symmetric_difference(qr_idx)
        db_idx = np.asarray(list(db_idx))
        qr_chunks = np.array_split(qr_idx, len(qr_idx) / chunk_size)
        db_idx_dict = {f: en for en, f in enumerate(db_idx)}

        result_dict['qr_idx'] = qr_idx
        result_dict['db_idx'] = db_idx
        # Process BEN one batch at a time.
        db_vectors = torch.tensor(vectors[db_idx]).cuda()
        for i in tqdm(qr_chunks, desc='Retrieval'):
            with torch.no_grad():
                scores = torch.mm(torch.tensor(vectors[i]).cuda(), db_vectors.t())
                indices = torch.argsort(scores, dim=1, descending=True)
                indices = indices.cpu().numpy()

            label_collection = list([])
            for j in range(20, 120, 20):
                multilabel_dict = MultiLabelMetrics(indices, targets,
                                                    i, db_idx, j)
                label_collection.append(multilabel_dict())

            for en_f, f in enumerate(i):
                for k in key_list:
                    # Remove overlap between matches and query/db sets
                    if en_f == 0:
                        aux_dict[k] = []
                    aux = remove_overlap(gt[str(f)][k][:].astype(int),
                                         qr_idx)
                    aux = [db_idx_dict[f] for f in aux]
                    aux_dict[k].append(aux)

            # Remove images with no matches in hard mode
            keep_idxs = [en for en, f in enumerate(aux_dict['hard']) if len(f) > 0]
            aux_dict['hard'] = [aux_dict['hard'][f] for f in keep_idxs]
            aux_dict['medium'] = [aux_dict['medium'][f] for f in keep_idxs]
            aux_dict['easy'] = [aux_dict['easy'][f] for f in keep_idxs]
            for k in key_list:
                res = compute_map_and_print(indices.T[:, keep_idxs], aux_dict[k])
                result_dict[k]['map'] += res[0] / len(qr_chunks)
                result_dict[k]['prs'] += res[1] / len(qr_chunks)
            multilabel_k = {k: v for i in label_collection for k, v in
                            i.items()}
            for k in multilabel_k:
                if k not in result_dict.keys():
                    result_dict[k] = 0.0
                result_dict[k] += multilabel_k[k] / len(qr_chunks)

        return {**result_dict}


def ucm_retrieval(retrieval_conf, vectors, targets, mode='test'):
    """
    Perform retrieval on the smaller datasets. Difficulty thresholds should
    be modified is not using the standard (0.6, 0.4, 0.2) values.
    :param retrieval_conf: Retrieval configuration.
    :param vectors: Matrix of image embeddings.
    :param targets: One-hot encoded target mutli-labels.
    :param mode: Train or test. Kept for compatibility.
    :return:
    Dictionary of performance metrics.
    """

    np.random.seed(retrieval_conf['seed'])
    key_list = [('easy', THRESHOLDS[0]),
                ('medium', THRESHOLDS[1]),
                ('hard', THRESHOLDS[2])]
    auxiliary_dict = dict({})
    result_dict = dict({})
    # Normalize
    vectors = np.asarray(vectors)
    vectors = vectors / np.expand_dims(np.linalg.norm(vectors, axis=1), axis=1)
    img_count = len(vectors)
    # Select query and database
    qr_idx = np.sort(np.random.choice(range(img_count), int(img_count * 0.2), replace=False))
    db_idx = set(range(img_count)).symmetric_difference(qr_idx)
    db_idx = np.asarray(list(db_idx))

    result_dict['qr_idx'] = qr_idx
    result_dict['db_idx'] = db_idx
    label_distance = cdist(targets[qr_idx], targets[db_idx], jaccard)
    for k, thresh in key_list:
        result_dict[k] = dict()
        for m in ['map', 'precision']:
            result_dict[k][m] = 0.0
        is_match = label_distance <= thresh
        auxiliary_dict[k] = [np.where(i)[0] for i in is_match]

    keep_idxs = [en for en, f in enumerate(auxiliary_dict['hard']) if len(f) > 0]
    auxiliary_dict['hard'] = [auxiliary_dict['hard'][f] for f in keep_idxs]
    auxiliary_dict['medium'] = [auxiliary_dict['medium'][f] for f in keep_idxs]
    auxiliary_dict['easy'] = [auxiliary_dict['easy'][f] for f in keep_idxs]
    # Compute distances
    feature_distances = np.dot(vectors[qr_idx], vectors[db_idx].T)
    top_indices = np.argsort(-feature_distances, axis=1)
    result_dict['ranks'] = top_indices
    label_collection = list([])
    for i in range(20, 120, 20):
        multilabel_dict = MultiLabelMetrics(top_indices, targets, qr_idx, db_idx, i)
        label_collection.append(multilabel_dict())

    for k, _ in key_list:
        res = compute_map_and_print(top_indices.T[:, keep_idxs], auxiliary_dict[k])
        result_dict[k]['map'] += res[0]
        result_dict[k]['precision'] += res[1]
    multilabel_k = {k: v for i in label_collection for k, v in i.items()}
    return {**result_dict, **multilabel_k}


def evaluate_retrieval(retrieval_conf, vectors, targets, mode='test', dataset='BigEarthNet'):
    """
    For small datasets the ranking is computed directly whereas for BEN this
    process requires batching.
    :param retrieval_conf: Retrieval configuration.
    :param vectors: Matrix of image embeddings.
    :param targets: One-hot encoded target mutli-labels.
    :param mode: Train or test.
    :param dataset: Name of the dataset.
    :return:
    Dictionary of performance metrics.
    """
    if dataset == 'BigEarthNet':
        return ben_retrieval(retrieval_conf, vectors, targets, mode)
    else:
        return ucm_retrieval(retrieval_conf, vectors, targets, mode)


def evaluate_ranks(ranks, targets, qr_idx, db_idx):
    """
    Evaluates performance based on a matrix of ranks instead of computing
    the ranking directly.
    :param retrieval_conf: Retrieval configuration.
    :param vectors: Matrix of image embeddings.
    :param targets: One-hot encoded target mutli-labels.
    :param mode: Train or test. Kept for compatibility.
    :return:
    Dictionary of performance metrics.
    """
    key_list = [('easy', THRESHOLDS[0]),
                ('medium', THRESHOLDS[1]),
                ('hard', THRESHOLDS[2])]
    auxiliary_dict = dict({})
    result_dict = dict({})

    label_distance = cdist(targets[qr_idx], targets[db_idx], jaccard)
    for k, thresh in key_list:
        result_dict[k] = dict()
        for m in ['map', 'precision']:
            result_dict[k][m] = 0.0
        is_match = label_distance <= thresh
        auxiliary_dict[k] = [np.where(i)[0] for i in is_match]

    keep_idxs = [en for en, f in enumerate(auxiliary_dict['hard']) if len(f) > 0]
    auxiliary_dict['hard'] = [auxiliary_dict['hard'][f] for f in keep_idxs]
    auxiliary_dict['medium'] = [auxiliary_dict['medium'][f] for f in keep_idxs]
    auxiliary_dict['easy'] = [auxiliary_dict['easy'][f] for f in keep_idxs]

    top_indices = ranks
    result_dict['ranks'] = top_indices
    label_collection = list([])
    for i in range(20, 120, 20):
        multilabel_dict = MultiLabelMetrics(top_indices, targets, qr_idx, db_idx, i)
        label_collection.append(multilabel_dict())

    for k, _ in key_list:
        res = compute_map_and_print(top_indices.T[:, keep_idxs], auxiliary_dict[k])
        result_dict[k]['map'] += res[0]
        result_dict[k]['precision'] += res[1]
    multilabel_k = {k: v for i in label_collection for k, v in i.items()}
    return {**result_dict, **multilabel_k}


def evaluate_ranks_ben(retrieval_conf, ranks, targets, qr_idx, db_idx):
    """
    Evaluates performance based on a matrix of ranks instead of computing
    the ranking directly. BEN requires batching of the process due
    to its large size.
    :param retrieval_conf: Retrieval configuration.
    :param vectors: Matrix of image embeddings.
    :param targets: One-hot encoded target mutli-labels.
    :param mode: Train or test. Kept for compatibility.
    :return:
    Dictionary of performance metrics.
    """
    chunk_size = retrieval_conf['chunk_size']
    np.random.seed(retrieval_conf['seed'])
    key_list = ['easy', 'medium', 'hard']
    aux_dict = dict()
    result_dict = dict()
    for k in key_list:
        result_dict[k] = dict()
        for m in ['map', 'prs']:
            result_dict[k][m] = 0.0
        for m in ['ACG@100', 'nDCG@100', 'wAP@100']:
            result_dict[m] = 0.0

    with h5py.File(retrieval_conf['path_gt'], 'r') as root:
        gt = root['test']

        qr_chunks = np.array_split(qr_idx, len(qr_idx) / chunk_size)
        idx_chunks = np.array_split(range(len(qr_idx)), len(qr_idx) / chunk_size)
        db_idx_dict = {f: en for en, f in enumerate(db_idx)}

        result_dict['qr_idx'] = qr_idx
        result_dict['db_idx'] = db_idx

        for en_i, i in enumerate(tqdm(qr_chunks, desc='Retrieval')):
            indices = ranks[idx_chunks[en_i]]

            label_collection = list([])
            for j in range(20, 120, 20):
                multilabel_dict = MultiLabelMetrics(indices, targets,
                                                    i, db_idx, j)
                label_collection.append(multilabel_dict())

            for en_f, f in enumerate(i):
                for k in key_list:
                    # Remove overlap between matches and query/db sets
                    if en_f == 0:
                        aux_dict[k] = []
                    aux = remove_overlap(gt[str(f)][k][:].astype(int),
                                         qr_idx)
                    aux = [db_idx_dict[f] for f in aux]
                    aux_dict[k].append(aux)

            # Remove images with no matches in hard mode
            keep_idxs = [en for en, f in enumerate(aux_dict['hard']) if
                         len(f) > 0]
            aux_dict['hard'] = [aux_dict['hard'][f] for f in keep_idxs]
            aux_dict['medium'] = [aux_dict['medium'][f] for f in keep_idxs]
            aux_dict['easy'] = [aux_dict['easy'][f] for f in keep_idxs]
            for k in key_list:
                res = compute_map_and_print(indices.T[:, keep_idxs],
                                            aux_dict[k])
                result_dict[k]['map'] += res[0] / len(qr_chunks)
                result_dict[k]['prs'] += res[1] / len(qr_chunks)
            multilabel_k = {k: v for i in label_collection for k, v in
                            i.items()}
            for k in multilabel_k:
                if k not in result_dict.keys():
                    result_dict[k] = 0.0
                result_dict[k] += multilabel_k[k] / len(qr_chunks)

        return {**result_dict}


def evaluate_for_visualization(retrieval_conf,
                               vectors,
                               targets,
                               n_queries=100,
                               top_k=10):
    """
    Perform retrieval and save a select number of queries/matches
    for visualization.
    :param retrieval_conf:
    :param vectors:
    :param targets:
    :param n_queries:
    :param top_k:
    :return:
    Dictionary with image indices and distances.
    """
    np.random.seed(retrieval_conf['seed'])
    res_dict = dict()

    vectors = np.asarray(vectors)
    vectors = vectors / np.expand_dims(np.linalg.norm(vectors, axis=1), axis=1)
    img_count = len(vectors)
    qr_idx = np.sort(np.random.choice(range(img_count), int(img_count*0.2), replace=False))
    db_idx = set(range(img_count)).symmetric_difference(qr_idx)
    qr_idx = qr_idx[:n_queries]
    db_idx = np.asarray(list(db_idx))
    qr_tgt = targets[qr_idx]

    scores = np.dot(vectors[qr_idx], vectors[db_idx].T)
    indices = np.argsort(-scores, axis=1).T
    indices = indices[:top_k, :n_queries]
    dist_top = np.zeros(indices.shape)
    for x in range(dist_top.shape[0]):
        for y in range(dist_top.shape[1]):
            db_tgt = targets[db_idx[indices[x, y]]]
            dist_top[x, y] = 1-jaccard(qr_tgt[y], db_tgt)

    res_dict['indices'] = indices
    res_dict['qr_idx'] = qr_idx
    res_dict['db_idx'] = db_idx
    res_dict['dist_top'] = dist_top

    return res_dict