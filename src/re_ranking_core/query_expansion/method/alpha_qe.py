import numpy as np
from .qe_base import QeBase
from .utils import compute_sim, sim_ranks
from tqdm import tqdm

class AlphaQueryExpansion(QeBase):
    def __init__(self, vectors, ranking, config):
        """
        Class for Average Query Expansion.
        :param vectors: Image embeddings.
        :param ranking: Dictinary containing ranks and query/database indices.
        :param config: Configuration dictionary.
        """
        super(AlphaQueryExpansion, self).__init__(vectors, ranking, config)
        self.k = config['aqe']['k']
        self.alpha = config['aqe']['alpha']
        self.query = self.vectors[ranking['qr_idx']]
        self.database = self.vectors[ranking['db_idx']]
    def expand_query(self, query, db_indices):
        """
        Expands a query with descriptors from database by weighting each
        database vector with its distance of each query to the database.
        :param query: Single query vector.
        :param db_indices: Indices of database vectors to expand query.
        :return:
        The expanded query vector.
        """
        renewed_query = query
        renewed_query = renewed_query / np.linalg.norm(renewed_query)

        for db_id in db_indices:
            db_vec = self.database[db_id, :]
            db_vec = db_vec / np.linalg.norm(db_vec)
            renewed_query += db_vec * np.power(compute_sim(query, db_vec), self.alpha)
        renewed_query = renewed_query / np.linalg.norm(renewed_query)
        return renewed_query


    def renew_query(self, q):
        """
        Create then new query representation.
        :param q: Query.
        :return:
        Expanded query embedding.
        """
        top_k_ranks = sim_ranks(query=q, database=self.database)
        top_k_ranks = top_k_ranks[0:self.k]
        renewed_q = self.expand_query(q, top_k_ranks).squeeze()
        return renewed_q


    def get_renewed_queries(self):
        """
        Process all the queries and generate new representations.
        :return:
        Expanded queries.
        """
        renewed_queries = np.zeros(shape=self.query.shape)
        for i, q in enumerate(self.query):
            renewed_queries[i, :] = self.renew_query(q)
        return renewed_queries

    def execute(self):
        ranks = np.zeros(shape=(self.query.shape[0], self.database.shape[0]))
        for i, q in enumerate(tqdm(self.query)):
            renewed_q = self.renew_query(q)
            updated_ranks = sim_ranks(query=renewed_q, database=self.database)
            ranks[i, :] = updated_ranks
        return ranks.astype(int)
