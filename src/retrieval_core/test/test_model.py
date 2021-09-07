import os
import torch
import numpy as np
from .extract_vectors import ExtractVectors
from src.common.utils import get_device
from src.common.evaluation.evaluate import Evaluate
from src.retrieval_core.models.model_factory import ModelFactory
from src.common.hyper_config import valid_datasets
from src.retrieval_core.models.checkpoint_util import CheckpointUtils
from src.common.utils import pretty_print


class TestModel:
    def __init__(self, config, tb_writer=None, n_iter=None, ckpt_dir=None):
        """
        Manages and instances all objects necessary for evaluation based on
        the configuration file.
        :param config: Configuration dictionary.
        :param tb_writer: Tensorboard writer.
        :param n_iter: Iteration number (only when training).
        :param ckpt_dir: Path to restore checkpoint from.
        """
        super(TestModel, self).__init__()

        self.config = config
        self.root_dir = config['root_dir']
        self.tb_writer = tb_writer
        self.iter = n_iter
        self.device = get_device(self.config['device'])

        self.mode = self.config['mode']
        self.test_config = config['test']
        self.dataset = self.test_config['dataset']
        self.augment_config = config['augment_config']
        self.batch_size = self.test_config['batch_size']
        self.nomenclature = self.test_config['nomenclature']
        self.ckpt_dir = ckpt_dir if ckpt_dir else self.test_config['ckpt_directory']
        self.global_multi_scale = self.test_config['global']['multi_scale']
        self.save = self.test_config['save']
        self.save_path = self.test_config['save_path']

        self.valid_datasets = valid_datasets

        self.config = config
        self.tb_writer = tb_writer
        self.iter = n_iter
        self.device = get_device(self.config['device'])
        self._gpu_id = config['device']

    def _extract(self, net):
        scale = self.global_multi_scale
        extractor = ExtractVectors(
            dataset=self.dataset,
            mode=self.mode,
            network=net,
            data_root=self.root_dir,
            config=self.config,
            device=self.device,
            batch_size=self.batch_size,
            multi_scale=scale
        )
        return extractor.extract_vectors()

    def _ckpt_dir(self):
        """Load latest checkpoint"""
        train_conf = self.config['train']
        model_conf = self.config['model']
        ckpt_name = "{}_{}_{}_{}_{}_{}".format(
            train_conf['dataset'],
            train_conf['learning_rate'],
            model_conf['backbone'],
            model_conf['pooling'],
            model_conf['module'],
            self.config['experiment']
        )
        return ckpt_name

    def init_model(self):
        model_factory = ModelFactory(self.config)
        model = model_factory.get_model()
        if self.mode == 'train' or self.mode == 'val':
            ckpt_loader = CheckpointUtils(self.ckpt_dir)
        else:
            ckpt_loader = CheckpointUtils(os.path.join(self.ckpt_dir, self._ckpt_dir()))
        ckpt = ckpt_loader.get_latest_ckpt_path()
        model.load_state_dict(
            torch.load(ckpt, map_location=self.device['unit'])['model_state_dict']
        )
        model.to(self.device['unit'])
        return model

    def run_inference(self, dataset, net):
        """
        Performs inference with a model.
        :param dataset: Dataset name.
        :param net: Trained model.
        :return:
        Query and database embeddings.
        """
        if dataset in self.valid_datasets:
            vectors, predictions, targets = self._extract(net)
        else:
            raise ValueError('Wrong dataset!')
        return vectors, predictions, targets

    def tensorboard_writer(self, info):
        """
        Writes metrics to tensorboard.
        info: Metric dictionary.
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('mAP/easy', info['easy']['map'], self.iter)
            self.tb_writer.add_scalar('mAP/medium', info['medium']['map'], self.iter)
            self.tb_writer.add_scalar('mAP/hard', info['hard']['map'], self.iter)
            self.tb_writer.add_scalar('ACG', info['ACG@100'], self.iter)
            self.tb_writer.add_scalar('nDCG', info['nDCG@100'], self.iter)
            self.tb_writer.add_scalar('wAP', info['wAP@100'], self.iter)

    def run_validation(self):

        model = self.init_model()
        vectors, predictions, targets = self.run_inference(self.dataset, model)
        if self.save:
            save_dict = dict()
            save_dict['vectors'] = vectors
            save_dict['predictions'] = predictions
            save_dict['targets'] = targets
            save_name = os.path.join(self.save_path, 'validation'+self._ckpt_dir())
            np.save(save_name, save_dict)
        results = Evaluate(
            dataset=self.dataset,
            vectors=vectors,
            logits=predictions,
            targets=targets,
            retrieval_conf=self.test_config,
            mode=self.mode,
            nomenclature=self.nomenclature
        ).run()
        self.tensorboard_writer(results)

    def run_evaluation(self):
        """
        Runs inference and evaluation, writes output to tensorboard if training.
        """
        model = self.init_model()
        vectors, predictions, targets = self.run_inference(self.dataset, model)
        if self.save:
            save_dict = dict()
            save_dict['vectors'] = vectors
            save_dict['predictions'] = predictions
            save_dict['targets'] = targets
            save_name = os.path.join(self.save_path, self._ckpt_dir())
            np.save(save_name, save_dict)

        results = Evaluate(
            dataset=self.dataset,
            vectors=vectors,
            logits=predictions,
            targets=targets,
            retrieval_conf=self.test_config,
            mode=self.mode,
            nomenclature=self.nomenclature
        ).run()
        pretty_print(results)

    def run_visualization(self):
        """
        Evaluates and saves the results as images.
        """
        save_name = os.path.join(self.save_path, self._ckpt_dir() + '.npy')
        if not os.path.exists(save_name):
            print('Extracting descriptors')
            model = self.init_model()
            vectors, predictions, targets = self.run_inference(
                self.dataset, model
            )
            if self.save:
                save_dict = dict()
                save_dict['vectors'] = vectors
                save_dict['predictions'] = predictions
                save_dict['targets'] = targets
                save_name = os.path.join(self.save_path, self._ckpt_dir())
                np.save(save_name, save_dict)
                save_name += '.npy'
        save_dict = np.load(save_name, allow_pickle=True).item()
        scale = self.global_multi_scale
        extractor = ExtractVectors(
            dataset=self.dataset,
            mode='test',
            network=None,
            data_root=self.root_dir,
            config=self.config,
            device=self.device,
            batch_size=self.batch_size,
            multi_scale=scale
        )
        evaluate = Evaluate(
            dataset=self.dataset,
            vectors=save_dict['vectors'],
            logits=save_dict['predictions'],
            targets=save_dict['targets'],
            retrieval_conf=self.test_config,
            mode=self.mode,
            nomenclature=self.nomenclature
        )
        evaluate.visualize(
            data_loader=extractor.dataset_loader(),
            save_path=os.path.join(self.save_path, self._ckpt_dir())
        )



