from src.base import Base
from src.common.io.yaml_io import yaml_reader
from src.retrieval_core.training.deploy_training import DeployTraining
from src.retrieval_core.test.test_model import TestModel


class DeployRetrieval(Base):
    def __init__(self, config_path):
        super(DeployRetrieval, self).__init__(config_path)
        self.config_path = config_path
        self.config = yaml_reader(self.config_path)
        self.mode = self.config['mode']

    def _train(self):
        DeployTraining(self.config).deploy()

    def _test(self):
        test_model = TestModel(self.config)
        test_model.run_evaluation()

    def _visualize(self):
        test_model = TestModel(self.config)
        test_model.run_visualization()

    def deploy(self):
        mode_map = {
            'train': self._train,
            'test': self._test,
            'visualize': self._visualize
        }
        if self.mode not in mode_map:
            raise ValueError('Unknown execution mode: {}'.format(self.mode))
        else:
            return mode_map[self.mode]()

