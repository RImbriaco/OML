import numpy as np
from typing import Tuple, Dict


class MultiLabelMetrics:
    def __init__(
            self,
            top_k_ranks: np.array,
            dataset_labels: np.array,
            qr_idxs: np.array,
            db_idxs: np.array,
            k: int = 100):
        """
        This class contains and computes all the multi-label metrics.
        :param top_k_ranks: Top-ranked matches
        :param dataset_labels: One-hot encoded labels
        :param qr_idxs: Query indices
        :param db_idxs: Database indices
        :param k: Value for which to compute metric@k (e.g. ndcg@100)
        """
        super(MultiLabelMetrics, self).__init__()
        self.k = k
        self.qr_idxs = qr_idxs
        self.db_idxs = db_idxs
        self.common_labels = self.get_common_labels(top_k_ranks, dataset_labels)

    def __len__(self) -> int:
        return self.k

    def get_labels(
            self,
            top_k_ranks: np.array,
            dataset_labels: np.array
    ) -> Tuple[np.array, np.array]:
        """
        Returns labels of query and top-k matches
        """
        truncated_ranks = top_k_ranks[:, :self.k]
        query_labels = dataset_labels[self.qr_idxs]
        rank_labels = dataset_labels[self.db_idxs][truncated_ranks]
        return query_labels, rank_labels

    def get_common_labels(
            self,
            top_k_ranks: np.array,
            dataset_labels: np.array
    ) -> np.array:
        """
        Returns number of common labels between query and top-k matches
        """
        query_labels, rank_labels = self.get_labels(top_k_ranks, dataset_labels)
        # All labels in top-k are now active
        query_labels = np.repeat(query_labels[:, np.newaxis, :], self.k, axis=1)
        common_labels = np.logical_and(query_labels, rank_labels)
        return common_labels

    def average_cg(self) -> np.array:
        """
        Compute Average Cumulative Gain
        """
        num_common_labels = np.sum(self.common_labels, axis=(1, 2))
        query_acg = num_common_labels / self.k  # per query
        return np.mean(query_acg)

    def compute_dcg(self, labels) -> np.array:
        """
        Compute Discounted Cumulative Gain
        """
        dcg = (np.power(2, labels) - 1) / np.log2(1 + np.arange(1, self.k + 1))
        return np.sum(dcg, axis=1)


    def normalized_discounted_cg(self, eps: float = 1e-6) -> np.array:
        """
        Compute Normalized Discounted Cumulative Gain
        """
        common_labels_per_rank = np.sum(self.common_labels, axis=2)  # per rank
        dcg = self.compute_dcg(common_labels_per_rank)
        idcg = self.compute_dcg(np.sort(-common_labels_per_rank) * -1) + eps
        return np.mean(dcg / idcg)

    @staticmethod
    def indicator_function(array: np.array):
        return np.where(array > 0, 1, 0)

    def weighted_ap(self, eps: float = 1e-6) -> np.array:
        """
        Compute Weighted Average Precision
        """
        common_labels_per_rank = np.sum(self.common_labels, axis=2)
        # ACG computation per rank for each query
        acg_per_rank = np.cumsum(common_labels_per_rank, axis=1) / (np.arange(1, self.k + 1))
        indicator = self.indicator_function(common_labels_per_rank)
        active_labels = acg_per_rank * indicator
        active_per_query = np.sum(active_labels, axis=1)
        # Not sure here confirm with Clint
        relevant_per_query = np.sum(indicator, axis=1) + eps
        return np.mean(active_per_query / relevant_per_query)

    def __call__(self) -> Dict[str, np.ndarray]:
        return {
            'ACG@'+str(self.k): self.average_cg(),
            'nDCG@'+str(self.k): self.normalized_discounted_cg(),
            'wAP@'+str(self.k): self.weighted_ap()
        }


