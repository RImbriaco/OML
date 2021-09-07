from src.retrieval_core.models.build_model import BuildModel


class ModelFactory:
    def __init__(self, config):
        """
        Factory to create an instance of a model.
        :param config: retrieval config file.
        """
        super(ModelFactory, self).__init__()

        # Base configurations
        self._config = config
        self._model_config = config['model']

        # Model configurations
        self._backbone = self._model_config['backbone']
        self._output_stride = self._model_config['output_stride']
        self._module = self._model_config['module']
        self._pool = self._model_config['pooling']
        self._inter_dim = self._model_config['part_dim']
        self._parts = self._model_config['parts']
        self._num_classes = self._model_config['num_classes']
        self._in_channels = self._model_config['channels']
        # Pre-trained settings
        self._pretrained = self._model_config['pretrained']


    def get_model(self):
        return BuildModel(
            self._backbone,
            self._pretrained,
            self._output_stride,
            self._module,
            self._parts,
            self._pool,
            self._inter_dim,
            self._num_classes,
            False,
            self._in_channels
        )