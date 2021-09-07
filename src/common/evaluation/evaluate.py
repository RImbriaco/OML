from src.common.evaluation.image_retrieval import evaluate_for_visualization
from src.common.evaluation.image_retrieval import evaluate_retrieval, \
    evaluate_ranks, evaluate_ranks_ben
from src.common.hyper_config import valid_datasets
from src.retrieval_core.visualize.viz import save_matches


class Evaluate:
    def __init__(self,
                 dataset,
                 vectors,
                 logits,
                 targets,
                 retrieval_conf,
                 mode,
                 nomenclature='new_classes'):
        """
        Evaluation class. Manages the retrieval and re-ranking processes
        as well as visualization. Outputs are printed in the console (retrieval)
        or saved to disk (viz).
        :param dataset: Dataset name.
        :param vectors: Extracted embedding matrix.
        :param logits: Class-prediction probability. Deprecated.
        :param targets: One-hot encoded targets.
        :param retrieval_conf: Configuration dictionary obtained from YAML.
        :param mode: Indicates whether training or testing.
        :param nomenclature: Indicates which class nomenclature (43-class or
        19-class to use). Old nomenclature (43-class has not been tested).
        """

        self.dataset = dataset
        self.vectors = vectors
        self.logits = logits
        self.targets = targets
        self.retrieval_conf = retrieval_conf['retrieval']
        self.visualize_config = retrieval_conf['visualization']
        self.mode = mode
        self.nomenclature = nomenclature
        self._valid_datasets = valid_datasets

    def rerank(self, ranks, qr_idx, db_idx):
        """
        Compute performance metrics based on the outputs of the re-ranking
        process.
        :param ranks: A matrix indicating the ranks of each database image in
        relation to the selected queries.
        :param qr_idx: Indices of the query images.
        :param db_idx: Indices of the database images.
        :return:
        Dictionary containing the performance metrics at various thresholds.
        """
        if self.dataset in self._valid_datasets:
            if self.dataset == 'BigEarthNet':
                result_dict = evaluate_ranks_ben(self.retrieval_conf,
                                                 ranks,
                                                 self.targets,
                                                 qr_idx,
                                                 db_idx)
            else:
                result_dict = evaluate_ranks(ranks,
                                             self.targets,
                                             qr_idx,
                                             db_idx)

            return result_dict
        else:
            raise ValueError('Dataset not compatible for evaluation')

    def run(self):
        """
        Perform ranking by computing the query-database descriptor distance
        and sorting it.
        :return:
        Dictionary containing the performance metrics at various thresholds.
        """
        if self.dataset in self._valid_datasets:
            retrieval = evaluate_retrieval(
                retrieval_conf=self.retrieval_conf,
                vectors=self.vectors,
                targets=self.targets,
                mode=self.mode,
                dataset=self.dataset
            )
            return retrieval
        else:
            raise ValueError('Dataset not compatible for evaluation')

    def visualize(self, data_loader, save_path):
        """
        Prepare queries and matches for visualization and save them to disk.
        :param data_loader: Dataloader object used for reading the images.
        :param save_path: Target path to write images to. Assigned to be
        'root/experiments/outputs/..'
        """
        if self.dataset in self._valid_datasets:
            retrieval = evaluate_for_visualization(
                retrieval_conf=self.retrieval_conf,
                vectors=self.vectors,
                targets=self.targets
            )
            save_matches(
                res_dict=retrieval,
                data_loader=data_loader,
                img_path=self.visualize_config['img_path'],
                save_path=save_path
            )
        else:
            raise ValueError('Dataset not compatible for evaluation')
