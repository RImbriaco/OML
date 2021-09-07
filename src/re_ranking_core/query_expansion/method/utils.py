import numpy as np


def compute_sim(query, database):
    """
    Compute the similarity matrix between query and database.
    :param query: Query embedding.
    :param database: Database embedding matrix.
    :return:
    Similarity matrix.
    """
    return np.dot(database, query.T)


def sim_ranks(query, database):
    """
    Compute the retrieval ranks between query and database.
    :param query: Query embedding.
    :param database: Database embedding matrix.
    :return:
    Retrieval rank matrix.
    """
    distance = compute_sim(query, database)
    return np.argsort(-distance, axis=0)