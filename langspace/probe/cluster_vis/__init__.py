from typing import Tuple, List, Iterable, Union
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from .. import LatentSpaceProbe
from .methods import ClusterVisualizationMethod as CvM
import math
import torch.nn.functional as F
import torch
from yellowbrick.text import TSNEVisualizer
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from yellowbrick.features import PCA
from yellowbrick.text import UMAPVisualizer

mpl.use('TkAgg')  # !IMPORTANT


class ClusterVisualizationProbe(LatentSpaceProbe):
    """
    Class for visualisation (PCA, T-SNE, UMAP) of the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable[Tuple[Union[str, Sentence], Union[str, int]]], sample_size: int,
                 methods: List[CvM], batch_size: int = 20):
        """
        Initialize the ClusterVisualizationProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (Iterable[Union[str, Sentence]]):
                [['animals require food for survival', 'arg0 v arg1 prp prp'], ..., ]

            sample_size (int): The number of data points to use for probing.
            methods (List[ClusterVisualizationMethod]): A list of visualisation methods to display.
            gen_factors (dict): The generative factors to probe with.
        """
        super(ClusterVisualizationProbe, self).__init__(model, data, sample_size)

        self.method = methods
        self.batch_size = batch_size

    @staticmethod
    def structure_viz(viz_list, sample_size=1000, TopK=5):
        """
        semantic role structure visualization
        only show the structure (remove repeated semantic role for each sentence). E.g., ARG0 ARG0 ARG0 V ARG1 ARG1 -> ARG0 V ARG1
        arguments:
        viz_list = [[sent, semantic role labels], [], ..., []]
        sample_size
        TopK
        """
        final_viz_list = []
        for pair in viz_list:
            sent, label = pair[0], pair[1]
            unique_label = []
            for tkn in label.split(' '):
                if len(unique_label) == 0:
                    unique_label.append(tkn)
                else:
                    if unique_label[-1] == tkn:
                        continue
                    else:
                        unique_label.append(tkn)
            final_viz_list.append((sent, ' '.join(unique_label)))

        second_values = [sublist[1] for sublist in final_viz_list]

        # Count the occurrences of label in dataset and only choose TopK as target labels.
        count_dict = Counter(second_values)
        sorted_counts = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        target_label_dict = dict([(i[0], 0) for i in sorted_counts[:TopK]])

        # make the input list balance.
        target_viz_list = []
        for i in final_viz_list:
            if i[1] in target_label_dict and target_label_dict[i[1]] <= sample_size / TopK:
                target_viz_list.append(i)
                target_label_dict[i[1]] += 1

        return target_viz_list

    @staticmethod
    def role_content_viz(viz_list, target_role, sample_size=1000, TopK=5):
        # 1. count unique role-content
        role_content_dict = dict()
        target_viz_list = []
        for pair in viz_list:
            sents, labels = pair[0].split(' '), pair[1].split(' ')
            for idx, tkn in enumerate(sents):
                label = labels[idx]
                key = label + ' : ' + tkn
                if key not in target_role:
                    continue
                else:
                    target_viz_list.append((pair[0], key))

        return target_viz_list

    def encoding(self, data: List[str]):
        """
        Encodes the sentences

        Args:
            data (List[str]): sentences

        Returns:
            Tensor: Latent representation
        """
        seed = list(data)
        if (len(seed) < 2):
            seed.append("")

        encode_seed = self.model.decoder.tokenizer(seed, padding="max_length", truncation=True,
                                                   max_length=self.model.decoder.max_len, return_tensors='pt')
        encode_seed_oh = F.one_hot(encode_seed["input_ids"], num_classes=len(self.model.decoder.tokenizer.get_vocab())).to(torch.int8)
        with torch.no_grad():
            latent = self.model.encode_z(encode_seed_oh)

        return latent

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
                           for i in range(math.ceil(self.sample_size / self.batch_size))], desc="Encoding"):
            latent_all.append(self.encoding([d[0] for d in data_batch]))
            label_all.extend([d[1] for d in data_batch])

        latent_all = torch.cat(latent_all)
        classes = list(set(label_all))

        # Create the visualizer and draw the vectors
        figs = list()

        for method in self.method:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)

            if method == CvM.TSNE:
                tsne = TSNEVisualizer(ax=ax, decompose_by=2, decompose="svd", classes=classes)
                tsne.fit(latent_all.numpy(), np.array(label_all))
                tsne.show()
                fig.savefig(f"t_sne.png", dpi=500)
                figs.append(tsne)

            if method == CvM.UMAP:
                umap = UMAPVisualizer(ax=ax, decompose_by=2, decompose="svd", classes=classes)
                umap.fit(latent_all.numpy(), np.array(label_all))
                umap.show()
                fig.savefig(f"umap.png", dpi=500)
                figs.append(umap)

            if method == CvM.PCA:
                visualizer = PCA(scale=True, classes=classes)
                # convert label_all to int list.
                visualizer.fit_transform(latent_all.numpy(), np.array([classes.index(i) for i in label_all]))
                visualizer.show()
                fig.savefig(f"pca.png", dpi=500)
                figs.append(visualizer)

        return figs



