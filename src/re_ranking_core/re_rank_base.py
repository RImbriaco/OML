from abc import ABC, abstractmethod


class ReRankBase(ABC):
    def __init__(self, vectors, config):
        self.vectors = vectors
        self.config = config

    @abstractmethod
    def apply(self):
        pass
