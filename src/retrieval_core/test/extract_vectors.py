import torch
import numpy as np
from tqdm import tqdm
from .extraction_ops import extract_ss
from src.retrieval_core.dataloader.loaders.create_dataset import CreateDataset


class ExtractVectors:
    def __init__(self, dataset, mode, network, data_root, config, device,
                 batch_size, multi_scale, transform=False):
        """
        Performs all the extraction operations before retrieval.
        :param dataset: Dataset name.
        :param mode: Train/val/test.
        :param network: Trained model.
        :param data_root: Root path of the dataset.
        :param config: Configuration dictionary.
        :param device: GPU ID.
        :param batch_size: Number of images to process per batch.
        :param multi_scale: Deprecated.
        :param transform: Specifies if transforms are necessary.
        """
        super(ExtractVectors, self).__init__()

        self.config = config
        self.dataset = dataset
        self.mode = mode
        self.net = network
        self.data_root = data_root
        self.augment_config = config['augment_config']
        self.device = device
        self.transform = transform
        self.batch_size = batch_size
        self.ms = multi_scale

    def dataset_loader(self):
        """
        Creates the dataloader from dataset.
        :return:
        Dataloader object.
        """
        dataset = CreateDataset(self.config, self.dataset, self.mode)
        data_loader = dataset.deploy_data_loader(self.mode)
        return data_loader

    def extract_vectors(self):
        """
        Returns vectors extracted at a single scales as np array.
        """
        self.net.eval()
        data_loader = self.dataset_loader()

        with torch.no_grad():
            # Get part-based model splits
            vecs = []
            y_true = []
            predicted_probs = []
            for i, data in enumerate(tqdm(data_loader)):
                if self.net.encoder.in_channels == 3:
                    bands = data["bands10"][:, :3].to(self.device['unit'])
                    bands = torch.cat(
                        (
                            bands[:, 2].unsqueeze(1),
                            bands[:, 1].unsqueeze(1),
                            bands[:, 0].unsqueeze(1)
                        ), dim=1
                    )
                else:
                    bands = torch.cat(
                        (
                            data["bands10"],
                            data["bands20"]
                        ), dim=1).to(self.device['unit'])
                target = data["label"].to(self.device['unit'])
                logits, embeddings = self.net(bands)
                # Try except to allow for part-based extraction
                try:
                    embeddings = embeddings / torch.norm(embeddings, dim=1).unsqueeze(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                except AttributeError:
                    embeddings = extract_ss(logits, embeddings)
                    probs = torch.sigmoid(logits[0][1]).cpu().numpy()
                predicted_probs += list(probs)
                y_true += list(target.cpu().numpy())
                vecs += list(embeddings.cpu().numpy())
        predicted_probs = np.asarray(predicted_probs)
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)
        y_true = np.asarray(y_true)
        return vecs, y_predicted, y_true
