from typing import Tuple, List, Iterable, Union
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from .. import LatentSpaceProbe
from .methods import ClusterVisualizationMethod
import math
import torch.nn.functional as F
import torch
from yellowbrick.text import TSNEVisualizer
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # !IMPORTANT
import matplotlib.pyplot as plt
from tqdm import tqdm
from yellowbrick.features import PCA
from yellowbrick.text import UMAPVisualizer


class ClusterVisualizationProbe(LatentSpaceProbe):
    def __init__(self, model: LangVAE, data: Iterable[Tuple[Union[str, Sentence], Union[str, int]]], sample_size: int,
                 method: List, batch_size: int = 20):
        super(ClusterVisualizationProbe, self).__init__(model, data, sample_size)

        self.method = method
        self.batch_size = batch_size

    def encoding(self, data):
        """
        args: single sentence (string)
        return: latent (list)
        """
        seed = list(data)
        if (isinstance(seed[0], Sentence)):
            seed = [sent.surface for sent in data]
        if (len(seed) < 2):
            seed.append("")

        encode_seed = self.model.decoder.tokenizer(seed, padding="max_length", truncation=True,
                                                   max_length=self.model.decoder.max_len, return_tensors='pt')
        encode_seed_oh = F.one_hot(encode_seed["input_ids"],
                                   num_classes=len(self.model.decoder.tokenizer.get_vocab())).to(torch.int8)
        encoded = self.model.encoder(encode_seed_oh)
        mu = encoded["embedding"]
        std = encoded["log_covariance"]
        latent, eps = self.model._sample_gauss(mu, std)

        return mu, std, latent

    def report(self):
        """
        args:
            Inputs:
                list: [[s1, label], [s2, label], ..., [sn, label]]
                E.g., [["the appalachian mountains are a kind of mountain", label1], ["the appalachian mountains are a kind of mountain", label2]]
            Return:
                save image.png
        """
        latent_all, label_all = [], []
        for data_batch in tqdm([self.data[i * self.batch_size: (i + 1) * self.batch_size]
                           for i in range(math.ceil(self.sample_size / self.batch_size))]):
            latent_all.append(self.encoding([d[0] for d in data_batch]))
            label_all.extend([d[1] for d in data_batch])

        latent_all = torch.stack(label_all)
        classes = list(set(label_all))

        # Create the visualizer and draw the vectors

        for method in self.method:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)

            if method == 't-SNE':
                tsne = TSNEVisualizer(ax=ax, decompose_by=2, decompose="svd", classes=classes)
                tsne.fit(latent_all.numpy(), np.array(label_all))
                tsne.show()
                fig.savefig(f"t_sne.png", dpi=500)

            if method == 'UMAP':
                umap = UMAPVisualizer(ax=ax, decompose_by=2, decompose="svd", classes=classes)
                umap.fit(latent_all.numpy(), np.array(label_all))
                umap.show()
                fig.savefig(f"umap.png", dpi=500)

            if method == 'PCA':
                visualizer = PCA(scale=True, classes=classes)
                # convert label_all to int list.
                visualizer.fit_transform(latent_all.numpy(), np.array([classes.index(i) for i in label_all]))
                visualizer.show()
                fig.savefig(f"pca.png", dpi=500)



