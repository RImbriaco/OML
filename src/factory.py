from src.retrieval_core.deploy_retrieval import DeployRetrieval
from src.re_ranking_core.deploy_re_rank import DeployReRank


class MethodFactory:
    def __init__(self, retrieval_config, rerank_config):

        self.retrieval_config = retrieval_config
        self.rerank_config = rerank_config

    def run_retrieval(self):
        if self.retrieval_config is not None:
            DeployRetrieval(self.retrieval_config).deploy()

    def run_rerank(self):
        if self.rerank_config is not None:
            DeployReRank(self.rerank_config).deploy()

    def deploy(self):
        self.run_retrieval()
        self.run_rerank()
        if self.retrieval_config is None and self.rerank_config is None:
            print('No configs passed!')



