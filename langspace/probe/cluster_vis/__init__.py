from typing import Tuple, List, Dict, Iterable, Union
from collections import Counter
from saf import Sentence
from langvae import LangVAE
from .. import LatentSpaceProbe
from .methods import ClusterVisualizationMethod as CvM
import math
import torch
from yellowbrick.text import TSNEVisualizer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from yellowbrick.features import PCA
from yellowbrick.text import UMAPVisualizer

# mpl.use('TkAgg')  # !IMPORTANT


class ClusterVisualizationProbe(LatentSpaceProbe):
    """
    Class for visualisation (PCA, T-SNE, UMAP) of the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable[Sentence], sample_size: int, target_roles: Dict[str, List[str]],
                 methods: List[CvM], cluster_annotation: str, batch_size: int = 20,
                 annotations: Dict[str, List[str]] = None, plot_label_map: Dict[str, str] = None):
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
        self.target_roles = target_roles
        self.cluster_annot = cluster_annotation
        self.annotations = annotations
        self.plot_label_map = plot_label_map

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
    def role_content_viz(viz_list: Iterable[Sentence], target_roles: Dict[str, List[str]],
                         annotation: str, plot_label_map: Dict[str, str]) -> List[Tuple[Sentence, str]]:
        target_viz_list = []
        label_map = plot_label_map or dict()
        for sent in viz_list:
            for idx, tok in enumerate(sent.tokens):
                if (tok.surface in target_roles.get(tok.annotations[annotation], [])):
                    label = tok.annotations[annotation]
                    key = f"{label_map.get(label, label)} : {tok.surface}"
                    target_viz_list.append((sent, key))

        return target_viz_list

    def report(self):
        """
        args:
            Inputs:
                list: [[s1, label], [s2, label], ..., [sn, label]]
                E.g., [["the appalachian mountains are a kind of mountain", label1], ["the appalachian mountains are a kind of mountain", label2]]
            Return:
                save image.png
        """
        target_viz_list = ClusterVisualizationProbe.role_content_viz(self.data, self.target_roles, self.cluster_annot,
                                                                     self.plot_label_map)

        latent_all, label_all = [], []
        for data_batch in tqdm([target_viz_list[i * self.batch_size: (i + 1) * self.batch_size]
                                for i in range(math.ceil(self.sample_size / self.batch_size))], desc="Encoding"):
            data = [d[0] for d in data_batch]
            if (data):
                latent_all.append(self.batched_encoding(data, self.annotations, self.batch_size))
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
                tsne.fit(latent_all.cpu().numpy(), np.array(label_all))
                tsne.show()
                fig.savefig(f"t_sne.png", dpi=500)
                figs.append(tsne)

            if method == CvM.UMAP:
                umap = UMAPVisualizer(ax=ax, decompose_by=2, decompose="svd", classes=classes)
                umap.fit(latent_all.cpu().numpy(), np.array(label_all))
                umap.show()
                fig.savefig(f"umap.png", dpi=500)
                figs.append(umap)

            if method == CvM.PCA:
                visualizer = PCA(scale=True, classes=classes)
                # convert label_all to int list.
                visualizer.fit_transform(latent_all.cpu().numpy(), np.array([classes.index(i) for i in label_all]))
                visualizer.show()
                fig.savefig(f"pca.png", dpi=500)
                figs.append(visualizer)

        return figs



