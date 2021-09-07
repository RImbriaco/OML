from torch import nn
from .metric.triplet_hard import TripletHard
from .metric.contrastive_avg import ContrastiveAverage
from .metric.oml import OML
from .metric.triplet_average import TripletAverage
from .metric.utils import jaccard_mining


class LossFactory(nn.Module):
    def __init__(self, id_loss=None, metric_loss=None, weight=None, parts=None,
                 sim_threshold=0.5, mining=None):
        """
        Generates instances of the losses. Hybrid losses (classification+metric)
        are possible.
        :param id_loss: Name of the classification loss.
        :param metric_loss: Name of the metric loss.
        :param weight: Weighting factor per loss.
        :param parts: Splits of the input tensor.
        :param sim_threshold: Label similarity threshold for metric mining.
        :param mining: Mining function.
        """
        super(LossFactory, self).__init__()

        self.id_loss_map = {
            'xent': nn.CrossEntropyLoss,
            'ml_xent': nn.BCEWithLogitsLoss,
            None: lambda *args, **kwargs: None
        }
        self.metric_loss_map = {
            'contrastive_avg': ContrastiveAverage,
            'triplet_avg': TripletAverage,
            'triplet_hard': TripletHard,
            'oml': OML,
            None: lambda *args, **kwargs: None
        }
        self.weight = weight
        self.parts = parts
        self.id_loss = self.id_loss_map[id_loss]()
        self.metric_loss = self.metric_loss_map[metric_loss]()
        self.sim_threshold = sim_threshold
        self.mining = mining

    def to(self, device):
        if self.id_loss is not None:
            self.id_loss.to(device)
        if self.metric_loss is not None:
            self.metric_loss.to(device)
        return self

    def compute_losses(self, embeddings, target, logits, metric=False):
        """
        Compute the losses independently.
        :param embeddings: Image mebeddings.
        :param target: Image labels.
        :param logits: Predicted classes.
        :param metric: Boolean if using metric losses.
        :return:
        Classification loss, metric loss
        """
        loss_metric = 0
        loss_id = self.id_loss(logits, target) if self.id_loss is not None else 0
        if metric:
            if isinstance(self.metric_loss, TripletHard):
                func_dict = {
                    'ji': jaccard_mining,
                }
                anchor, pos, neg = func_dict[self.mining](self.sim_threshold, target, embeddings)
                loss_metric = self.metric_loss(anchor, pos, neg)
            else:
                loss_metric = self.metric_loss(embeddings, target)

        return loss_id, loss_metric

    def branch_losses(self, branches, targets, metric=False):
        """
        If more than one split/branch is used, compute the loss per branch
        and sum it.
        :param branches: Input tensor splits.
        :param targets: One-hot encoded labels.
        :param metric: Boolean if using metric losses.
        :return:
        Sum of branch losses.
        """
        losses_id = 0
        losses_metric = 0
        for branch in branches:
            loss_id, loss_metric = self.compute_losses(
                embeddings=branch[0],
                target=targets,
                logits=branch[1],
                metric=metric
            )
            losses_id += loss_id
            losses_metric += loss_metric
        return losses_id / len(branches), losses_metric / len(branches)

    def single_branch_losses(self, logits, embeddings, targets, metric=False):
        """
        Special case when using no branches/splits.
        :param logits: Predictions.
        :param embeddings: Embeddings.
        :param targets: One-hot encoded labels.
        :param metric: Boolean if using metric losses.
        :return:
        Class and metric losses.
        """
        return self.compute_losses(embeddings, targets, logits, metric)

    def forward(self, local_branches, global_branches, target):
        """
        Compute the losses per branch/loss type and return their weighted sum.
        :param local_branches: Smaller image splits.
        :param global_branches: Global image split.
        :param target: One-hot encoded labels.
        :return:
        Total loss.
        """
        use_metric = False if self.metric_loss is None else True
        if len(self.parts) == 1 and self.parts[0] == [1, 1, 1]:
            g_id, g_mt = self.single_branch_losses(
                logits=local_branches,
                embeddings=global_branches,
                targets=target,
                metric=use_metric
            )
            total_loss = self.weight[0] * g_id + self.weight[1] * g_mt
            return total_loss, g_id, g_mt

        else:

            l_id, l_mt = self.branch_losses(local_branches, target, use_metric)
            g_id, g_mt = self.branch_losses(global_branches, target, use_metric)
            total_loss = self.weight[0] * (l_id + g_id) + \
                         self.weight[1] * (l_mt + g_mt)
            return total_loss, l_id + g_id, l_mt + g_mt
