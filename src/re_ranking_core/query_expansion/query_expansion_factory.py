from .method.average_qe import AverageQE
from .method.alpha_qe import AlphaQueryExpansion
from .method.db_expansion import JaccardAffinity
from src.re_ranking_core.re_rank_base import ReRankBase

QUERY_EXP_METHODS = {
    'qe': AverageQE,
    'aqe': AlphaQueryExpansion,
    'ja': JaccardAffinity,
}

class QueryExpansionFactory(ReRankBase):
    def __init__(self, method, vectors, ranking, config):
        """
        Base class for instancing re-ranking classes.
        :param method: Re-ranking method.
        :param vectors: Image embeddings.
        :param ranking: Dictinary containing ranks and query/database indices.
        :param config: Configuration dictionary.
        """
        super(QueryExpansionFactory, self).__init__(vectors, config)

        if method not in QUERY_EXP_METHODS.keys():
            raise ValueError('Method {} not implemented!'.format(method))
        else:
            self.method = QUERY_EXP_METHODS[method](vectors, ranking, config)

    def renewed_queries(self):
        return self.method.get_renewed_queries()

    def apply(self):
        return self.method.execute()
