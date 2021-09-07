import numpy as np
from .qe_base import QeBase
from .utils import sim_ranks
import src.re_ranking_core.query_expansion.utils as qe_utils
from src.re_ranking_core.query_expansion.utils import apply_qe


class AverageQE(QeBase):
    def __init__(self, vectors, ranking, config):
        """
        Class for Average Query Expansion.
        :param vectors: Image embeddings.
        :param ranking: Dictinary containing ranks and query/database indices.
        :param config: Configuration dictionary.
        """
        super(AverageQE, self).__init__(vectors, ranking, config)
        self.k = config['qe']['k']
        self.query = self.vectors[ranking['qr_idx']]
        self.database = self.vectors[ranking['db_idx']]


    def renew_query(self, q):
        """
        Create then new query representation.
        :param q: Query.
        :return:
        Expanded query embedding.
        """
        top_k_ranks = sim_ranks(query=q, database=self.database)
        top_k_ranks = top_k_ranks[0: self.k]
        return qe_utils.expand_query(q, self.database, top_k_ranks).squeeze()


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
        return apply_qe(self.query, self.database, self.k)



