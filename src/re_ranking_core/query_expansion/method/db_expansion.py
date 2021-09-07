import os
import numpy as np
from .qe_base import QeBase
from .utils import sim_ranks
import src.re_ranking_core.query_expansion.utils as qe_utils
import h5py
from tqdm import tqdm
import torch


class JaccardAffinity(QeBase):
    def __init__(self, vectors, ranking, config):
        """
        Class for Jaccard affinity re-ranking.
        :param vectors: Image embeddings.
        :param ranking: Dictinary containing ranks and query/database indices.
        :param config: Configuration dictionary.
        """
        super(JaccardAffinity, self).__init__(vectors, ranking, config)

        qe_config = config['csqe']
        self.k = qe_config['k']
        self.ks = qe_config['ks']
        self.kc = qe_config['kc']
        self.query = self.vectors[ranking['qr_idx']]
        self.database = self.vectors[ranking['db_idx']]
        if config['dataset'] == 'BigEarthNet':
            q = h5py.File(self.config['retrieval']['path_gt'], 'r')
            q_test = q['test']
            labels = [q_test[k]['tgt'][:] for k in tqdm(ranking['db_idx'].astype(str))]
            self.db_labels = np.array(labels)
        else:
            labels = qe_utils.labels_csv_to_array(qe_config['labels_csv_path'])
            self.db_labels = labels[ranking['db_idx']]
        self.sim_affinity_path = qe_config['sim_affinity_path']
        self.label_affinity_path = qe_config['label_affinity_path']
        self.precomputed_graph_path = qe_config['precomputed_graph_path']

    def label_affinity_matrix(self):
        """
        Creates the label affinity matrix.
        :return:
        Jaccard distance of each db vector to another.
        """
        if not os.path.exists(self.label_affinity_path):
            label_affinity = qe_utils.save_label_affinity(self.label_affinity_path,
                                                          self.db_labels)
        else:
            label_affinity = np.load(self.label_affinity_path)

        return label_affinity

    def sim_affinity_matrix(self):
        """
        Creates the similarity affinity matrix.
        :return:
        Feature distance of each db vector to another.
        """
        if not os.path.exists(self.sim_affinity_path):
            sim_affinity = qe_utils.save_sim_affinity(self.sim_affinity_path,
                                                      self.database)
        else:
            sim_affinity = np.load(self.sim_affinity_path)
        return sim_affinity

    def joint_affinity(self):
        """
        Composition of similarity and label affinities
        :return:
        Database-to-database ranks.
        """
        label_aff_mat = self.label_affinity_matrix()
        db_ranks = np.argsort(-label_aff_mat, axis=0)
        return db_ranks

    def get_renewed_queries(self):
        pass

    def torch_cdist(self, xa, xb):
        """
        Compute Jaccard distance between two collections using torch.
        :param xa: Embedding collection 1.
        :param xb: : Embedding collection 2.
        :return:
        Distance between elements in xa and xb.
        """
        xa = torch.tensor(xa).cuda()
        xb = torch.tensor(xb).cuda()
        distance = torch.zeros((xa.shape[0], xb.shape[0])).cuda()
        for i, label in enumerate(xa):
            l_and = torch.logical_and(label, xb)
            l_or = torch.logical_or(label, xb)
            jaccard_index = torch.sum(l_and, dim=1) /\
                            (torch.sum(l_or, dim=1) + 1e-6)
            distance[i] = jaccard_index
        return distance.cpu().numpy()

    def execute(self):
        if self.config['dataset'] == 'BigEarthNet':
            new_ranks = np.zeros((self.query.shape[0],
                                  self.db_labels.shape[0]), dtype=int).T
            db_vectors = torch.tensor(self.database).cuda()
            for i, qr in enumerate(tqdm(self.query, desc='DB Expansion')):
                with torch.no_grad():
                    score = torch.mm(torch.tensor(qr).cuda().unsqueeze(0),
                                       db_vectors.t())
                    indices = torch.argsort(-score).squeeze().cpu().numpy()
                    jaccard_index = self.torch_cdist(
                        np.expand_dims(
                            self.db_labels[indices[0]], 0),
                            self.db_labels)
                    new_ranks[:, i] = np.argsort(-jaccard_index).astype(int)
        else:
            ranks = sim_ranks(self.query, self.database)
            db_ranks = self.joint_affinity()
            new_ranks = db_ranks[:, ranks[0, :]]
        # Size_db x size_qr
        return new_ranks.T