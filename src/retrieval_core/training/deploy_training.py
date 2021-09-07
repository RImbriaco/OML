from src.common.hyper_config import valid_datasets
from src.retrieval_core.training.train_ben import Train


class DeployTraining:
    def __init__(self, config):
        """
        Trains model from given configurations
        Args:
            config: retrieval config file

        """
        super(DeployTraining, self).__init__()
        self._config = config
        self._dataset = config['train']['dataset']

    def deploy(self):
        if self._dataset in valid_datasets:
            Train(self._config)
        else:
            raise ValueError('Unknown dataset: {}'.format(self._dataset))


