import numpy as np
from .method.utils import sim_ranks, compute_sim
from tqdm import tqdm


def expand_query(query, database, db_indices):
    """
    Expands a query with descriptors from database
    :param query: a single query vector
    :param database: database vectors
    :param db_indices: indices of database vectors to expand query

    :return:
    An expanded query vector.
    """
    renewed_query = np.copy(query)
    for db_id in db_indices:
        renewed_query += database[db_id, :]
    renewed_query = renewed_query / np.linalg.norm(renewed_query)
    return renewed_query


def apply_qe(query, database, k):
    """
    Performs query expansion.
    :param query: a single query vector
    :param database: database vectors
    :param k: Top-k ranks to employ.
    :return:
    Re-ranked retrieval results.
    """
    ranks = np.zeros(shape=(query.shape[0], database.shape[0]))
    for i, q in enumerate(tqdm(query)):
        top_k_ranks = sim_ranks(query=q, database=database)
        top_k_ranks = top_k_ranks[0:k]
        renewed_q = expand_query(q, database, top_k_ranks).squeeze()
        ranks[i, :] = sim_ranks(query=renewed_q, database=database)
    return ranks.astype(int)


def labels_csv_to_array(csv_path):
    """
    Convert CSV files to usable labels.
    :param csv_path: Path to CSV.
    :return:
    Label list.
    """
    with open(csv_path) as f:
        text = f.readlines()
        img_list = [t.split(',') for t in text]
    img_list = np.array(img_list)
    img_labels = img_list[:, 1:].astype('int')
    return img_labels


def save_sim_affinity(save_path, database):
    """
    Store results in disk.
    :param save_path: Output path.
    :param database: Database embeddings.
    :return:
    Database-to-database similarities.
    """
    db_sim = compute_sim(database, database)
    np.save(save_path, db_sim)
    return db_sim


def save_label_affinity(save_path, labels, eps=1e-6):
    """
    Sotre label distance to disk.
    :param save_path: Output path.
    :param labels: Label matrix.
    :param eps: Epsilon.
    :return:
    Jaccard distance.
    """
    repeat_val = labels.shape[0]
    labels = np.expand_dims(labels, axis=1)
    labels = np.repeat(labels, repeat_val, axis=1)
    labels_transpose = np.transpose(labels, axes=(1, 0, 2))
    l_and = np.logical_and(labels, labels_transpose)
    l_or = np.logical_or(labels, labels_transpose)
    jaccard_dist = np.sum(l_and, axis=2) / (np.sum(l_or, axis=2) + eps)
    np.save(save_path, jaccard_dist)
    return jaccard_dist
