from src.base import Base
from src.common.io.yaml_io import yaml_reader
from src.re_ranking_core.re_rank_factory import ReRankFactory


class DeployReRank(Base):
    def __init__(self, config_path):
        super(DeployReRank, self).__init__(config_path)

        self.config_path = config_path

    def deploy(self):
        config = yaml_reader(self.config_path)
        re_rank_factory = ReRankFactory(config)
        re_rank_factory.apply()
