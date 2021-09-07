import os
import torch
from .optimizer import create_optimizer
from .losses.loss_factory import LossFactory

from src.common.utils import get_device, get_dirs, get_latest_ckpt, model_to_device
from src.retrieval_core.dataloader.loaders.create_dataset import CreateDataset
from src.retrieval_core.models.model_factory import ModelFactory
from src.retrieval_core.test.test_model import TestModel

from tensorboardX import SummaryWriter


class Train:
    def __init__(self, config):
        """
        Trains model from given configurations
        Args:
            config: retrieval config file

        """
        super(Train, self).__init__()

        # Base settings
        self._config = config
        self._mode = config['mode']
        self._root_dir = config['root_dir']
        self._model_cfg = config['model']
        self._train_config = config['train']
        self._augment_config = config['augment_config']
        self._dev = get_device(config['device'])
        self._gpu_id = config['device']

        # Logging settings
        self._d_step = config['display']['display_step']
        self._ckpt_step = config['train']['checkpoint_step']

        # Train settings
        self._nomeclature = config['train']['nomenclature']
        self._dataset = config['train']['dataset']
        self._resume = config['train']['resume']
        self._id_loss = config['train']['id_loss']
        self._metric_loss = config['train']['metric_loss']
        self._loss_weight = tuple(config['train']['loss_weight'])
        self._sim_threshold = config['train']['sim_threshold']
        self._mining = config['train']['mining_strategy']
        self._opt = config['train']['optimizer']
        self._up_iter = config['train']['update_iter']
        self._epochs = config['train']['epochs']
        self._lr = float(config['train']['learning_rate'])

        # Logging settings
        self._path_maps = get_dirs(config)
        self._sum_path = self._path_maps['summary']
        self._ckpt_path = self._path_maps['checkpoints']

        self.train_model()

    def _set_loss(self):
        return LossFactory(self._id_loss, self._metric_loss, self._loss_weight,
                           self._model_cfg['parts'], self._sim_threshold, self._mining
                           ).to(self._dev['unit'])

    def _set_optimizer(self, model):
        params = model.parameters()
        opt, scheduler = create_optimizer(params, self._opt, self._lr)
        return opt, scheduler

    def _writer(self):
        try:
            writer = SummaryWriter(log_dir=self._sum_path)
        except TypeError:
            writer = SummaryWriter(logdir=self._sum_path)
        return writer

    @staticmethod
    def tb_scalars(loss, id_loss, metric_loss, writer, n_iter):
        writer.add_scalar('total_loss', loss, n_iter)
        writer.add_scalar('id_loss', id_loss, n_iter)
        writer.add_scalar('metric_loss', metric_loss, n_iter)

    def _train_epoch(self, data_loader, model, loss_fn, opt, epoch, writer):
        """
        Trains a single epoch, displays logs

        Args:
            data_loader: data loader
            model: model
            loss_fn: loss function to apply
            opt: optimizer to use
            epoch: n^th epoch
            writer: tensorboard writer

        Returns:
            n_iter: number of iterations
        """

        def to_np(val):
            return val.data.cpu().numpy() if isinstance(val, torch.Tensor) else val

        def band_selection(val):
            if model.encoder.in_channels == 3:
                val = val["bands10"][:, :3].to(self._dev['unit'])
                val = torch.cat((val[:, 2].unsqueeze(1),
                                 val[:, 1].unsqueeze(1),
                                 val[:, 0].unsqueeze(1)), dim=1)
            else:
                val = torch.cat((val["bands10"], val["bands20"]),
                                dim=1).to(self._dev['unit'])
            return val

        model.train()

        n_iter = 0
        for i, data in enumerate(data_loader):
            bands = band_selection(data)
            target = data["label"].to(self._dev['unit'])
            opt.zero_grad()
            logits, embeddings = model(bands)
            loss, id_loss, metric_loss = loss_fn(logits, embeddings, target)
            loss.backward()
            opt.step()
            n_iter = (epoch - 1) * data_loader.__len__() + i

            if i % self._d_step == 0:
                print('Epoch: {} ,iter: {}, total_loss: {:.3f} '
                      'ID loss: {:.3f} metric_loss {:.3f}'.format(
                    epoch, i, to_np(loss), to_np(id_loss),
                    to_np(metric_loss)))
                self.tb_scalars(loss, id_loss, metric_loss, writer, n_iter)
        return n_iter

    def _resume_train(self, model, scheduler):
        """
        Start epoch is set to 0 if no resume is applied, otherwise start epoch
        is collected from the model state.
        Returns:
            start_epoch: the epoch to resume
        """
        start_epoch = 0
        if self._resume:
            state = torch.load(os.path.join(get_latest_ckpt(self._ckpt_path)))
            model.load_state_dict(state['model_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])
            start_epoch = state['epoch']
            print('Resuming training from epoch: {}'.format(start_epoch))
        return start_epoch

    def _eval_model(self, writer, n_it):
        test_model = TestModel(self._config, writer, n_it, self._ckpt_path)
        test_model.mode = 'val'
        test_model.run_validation()

    def _save_model(self, model, scheduler, ep, writer, n_iter):
        """
        Saves state of the model, scheduler and epoch.

        Args:
            model: model
            scheduler: scheduler
            ep: epoch count
            writer: tensor board writer
            n_iter:

        Returns:

        """
        if (ep + 1) % self._ckpt_step == 0:
            save_dict = {
                'epoch': (ep + 1),
                'model_state_dict': model.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': self._config
            }
            torch.save(save_dict, os.path.join(
                self._ckpt_path, 'epoch_{}.pth'.format(ep + 1)))

            self._eval_model(writer, n_iter)
            model.train()

    def train_model(self):
        """
        Trains model from scratch or resumed state and saves model.
        Returns:

        """
        writer = self._writer()

        # Dataset
        dataset = CreateDataset(self._config, self._dataset, self._mode)
        data_loader = dataset.deploy_data_loader('train')

        # Construct and initialize model
        model_factory = ModelFactory(self._config)
        model = model_factory.get_model()
        model = model_to_device(model, self._gpu_id)
        model.train()

        # Loss and optimizer
        loss_fn = self._set_loss()
        opt, scheduler = self._set_optimizer(model)

        # Load model with weights from previous ckpt, initialize start epoch.
        start_epoch = self._resume_train(model, scheduler)

        # Train and save model
        for ep in range(start_epoch, self._epochs):
            n_iter = self._train_epoch(
                data_loader, model, loss_fn, opt, ep + 1, writer)
            scheduler.step()
            self._save_model(model, scheduler, ep, writer, n_iter)
