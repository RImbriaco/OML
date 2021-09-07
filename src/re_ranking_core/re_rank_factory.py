import numpy as np
from .query_expansion.query_expansion_factory import QUERY_EXP_METHODS
from .query_expansion.query_expansion_factory import QueryExpansionFactory
from src.common.evaluation.evaluate import Evaluate
from src.common.utils import pretty_print

class ReRankFactory:
    def __init__(self, config):
        super(ReRankFactory, self).__init__()

        self.config = config
        self.rerank_scheme = config['rerank_scheme']
        self.scheme_count = len(self.rerank_scheme)
        np.random.seed(self.config['seed'])

        for rr_method in config['rerank_scheme']:
            if rr_method not in QUERY_EXP_METHODS.keys():
                raise ValueError('Method {} not implemented!'.format(rr_method))

    def unpack_data(self):
        data = np.load(self.config['path_to_desc'], allow_pickle=True).item()
        vectors, preds, targets = data.values()
        return np.array(vectors), preds, targets

    def prepare_evaluator(self):
        vectors, preds, targets = self.unpack_data()
        return Evaluate(dataset=self.config['dataset'],
                        vectors=vectors,
                        logits=preds,
                        targets=targets,
                        retrieval_conf=self.config,
                        mode='test')

    def plain_rank(self):
        """
        Performs plain ranking using the descriptors without any re-ranking
        techniques applied.
        Returns:
            ranks:
        """
        evaluator = self.prepare_evaluator()
        result_dict = evaluator.run()
        return result_dict

    def apply_method(self, method, plain_results):
        vectors, _, _ = self.unpack_data()
        evaluator = self.prepare_evaluator()
        new_ranks = QueryExpansionFactory(method=method,
                                          vectors=vectors,
                                          ranking=plain_results,
                                          config=self.config).apply()
        results = evaluator.rerank(ranks=new_ranks,
                                   qr_idx=plain_results['qr_idx'],
                                   db_idx=plain_results['db_idx'])
        return results

    def apply(self):
        if self.scheme_count == 0:
            results = self.plain_rank()
            print('Plain ranks.')
            pretty_print(results)
        else:
            plain_results = self.plain_rank()
            for method in self.rerank_scheme:
                results = self.apply_method(method, plain_results)
                print('Results with:', method)
                pretty_print(results)




