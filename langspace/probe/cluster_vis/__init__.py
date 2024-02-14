from typing import List, Iterable, Union
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from .. import LatentSpaceProbe
from .methods import ClusterVisualizationMethod
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
    def __init__(self, model: LangVAE, data: Iterable[Union[str, Sentence]], sample_size: int,
                 method: List):
        super(ClusterVisualizationProbe, self).__init__(model, data, sample_size)

        self.data = data
        self.sample_size = sample_size
        self.model = model
        self.method = method

    def encoding(self, data):
        """
        args: single sentence (string)
        return: latent (list)
        """
        seed = [data, data]
        encode_seed = self.model.decoder.tokenizer(seed, return_tensors='pt')
        encode_seed_oh = F.one_hot(encode_seed["input_ids"], num_classes=len(self.model.decoder.tokenizer.get_vocab())).to(torch.int8)
        encoded = self.model.encoder(encode_seed_oh)
        mu = encoded["embedding"][0]
        std = encoded["log_covariance"][0]
        latent, eps = self.model._sample_gauss(mu, std)
        return latent.tolist()

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
        for data in tqdm(self.data[:self.sample_size]):
            latent_all.append(self.encoding(data[0]))
            label_all.append(data[1])

        # Create the visualizer and draw the vectors
        for method in self.method:
            if method == 't-SNE':
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                classes = list(set(label_all))
                tsne = TSNEVisualizer(ax=ax, decompose_by=2, decompose="svd", classes=classes)
                tsne.fit(np.array(latent_all), np.array(label_all))
                tsne.show()
                fig.savefig(f"t_sne.png", dpi=500)

            if method == 'UMAP':
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                classes = list(set(label_all))
                umap = UMAPVisualizer(ax=ax, decompose_by=2, decompose="svd", classes=classes)
                umap.fit(np.array(latent_all), np.array(label_all))
                umap.show()
                fig.savefig(f"umap.png",dpi=500)

            if method == 'PCA':
                fig = plt.figure(figsize=(10, 10))
                classes = list(set(label_all))
                visualizer = PCA(scale=True, classes=classes)
                # convert label_all to int list.
                visualizer.fit_transform(np.array(latent_all), np.array([classes.index(i) for i in label_all]))
                visualizer.show()
                fig.savefig(f"pca.png",dpi=500)



