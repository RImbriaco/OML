import numpy as np
from abc import ABC, abstractmethod


class QeBase(ABC):
    def __init__(self, vectors, ranking, config):
        vectors = vectors / np.expand_dims(np.linalg.norm(vectors, axis=1),
                                           axis=1)
        self.vectors = vectors
        self.ranking = ranking
        self.config = config

    @abstractmethod
    def get_renewed_queries(self):
        pass

    @abstractmethod
    def execute(self):
        pass
