from abc import abstractmethod


class Base:
    def __init__(self, config_path):
        self.config_path = config_path

    @abstractmethod
    def deploy(self):
        pass
