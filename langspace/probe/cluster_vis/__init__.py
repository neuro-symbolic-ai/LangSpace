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
    A probe for visualizing the latent space of a language VAE via clustering techniques.

    This probe supports visualization methods including PCA, T-SNE, and UMAP. It processes a
    collection of sentences, extracts their latent representations, and generates visual plots
    highlighting clusters based on provided target roles and annotations. Generated plots are saved
    to image files.

    Attributes:
        model (LangVAE): The LM-VAE model whose latent space is to be analyzed.
        data (Iterable[Sentence]): An iterable of Sentence objects representing the input data.
        sample_size (int): The number of data points to process for visualization.
        target_roles (Dict[str, List[str]]): A mapping between annotation categories and target tokens
            for visualization clustering.
        method (List[ClusterVisualizationMethod]): A list of visualization methods to apply (e.g., TSNE, UMAP, PCA).
        cluster_annot (str): The annotation name used to filter or identify clusters.
        batch_size (int): The number of data points to encode in each batch.
        annotations (Dict[str, List[str]], optional): Optional dictionary of annotation types to be processed and all
        their possible values, for conditional encoding.
        plot_label_map (Dict[str, str], optional): Optional mapping to provide custom labels for plotting.
    """

    def __init__(self, model: LangVAE, data: Iterable[Sentence], sample_size: int, target_roles: Dict[str, List[str]],
                 methods: List[CvM], cluster_annotation: str, batch_size: int = 20,
                 annotations: Dict[str, List[str]] = None, plot_label_map: Dict[str, str] = None):
        """
        Initialize the ClusterVisualizationProbe with the specified model, data, and configuration options.

        Args:
            model (LangVAE): The language VAE whose latent space will be visualized.
            data (Iterable[Sentence]): An iterable containing Sentence objects as the data input.
            sample_size (int): The total number of data points to use for probing.
            target_roles (Dict[str, List[str]]): Dictionary mapping role names to lists of target tokens
                that define the roles for visualization.
            methods (List[ClusterVisualizationMethod]): List of visualization methods (enumerated in CvM) to be applied.
            cluster_annotation (str): The annotation key used to extract cluster labels from tokens.
            batch_size (int, optional): Batch size for encoding sentences. Default is 20.
            annotations (Dict[str, List[str]], optional): Optional dictionary of annotation types to be processed and all
            their possible values, for conditional encoding.
            plot_label_map (Dict[str, str], optional): Optional mapping for renaming labels in the plots.
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
        Generate a structured visualization list by removing consecutive duplicate semantic role labels.

        This method processes an input list of (sentence, semantic role labels) pairs. For each pair,
        it removes repeated adjacent role labels; for example, transforming "ARG0 ARG0 ARG0 V ARG1 ARG1"
        into "ARG0 V ARG1". It then counts the occurrences of each unique label pattern and selects
        only the top K most frequent labels. Finally, the input list is balanced to include only up to
        (sample_size / TopK) instances for each target label.

        Args:
            viz_list (List[Tuple[Sentence, str]]): A list of tuples where each tuple contains a sentence
                and a string of semantic role labels separated by spaces.
            sample_size (int, optional): The maximum number of data points to consider. Default is 1000.
            TopK (int, optional): The number of most frequent unique role structures to retain. Default is 5.

        Returns:
            A filtered and balanced list of (sentence, unique role structure) pairs.
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
        """
        Extract sentences and associate them with role-specific labels for content visualization.

        The method iterates through each sentence and examines its tokens. If a token's surface form
        is found in the list of target tokens (as specified by the given annotation in target_roles),
        it constructs a label. The label is either the original annotation or a remapped label as defined
        in plot_label_map. Each sentence with an associated label forms a tuple that is added to the resulting list.

        Args:
            viz_list (Iterable[Sentence]): An iterable of Sentence objects to be processed.
            target_roles (Dict[str, List[str]]): Dictionary mapping annotation keys to a list of target tokens.
            annotation (str): The key used to access the token's annotations for role filtering.
            plot_label_map (Dict[str, str]): Optional mapping to translate or reformat the original annotation label.

        Returns:
            List[Tuple[Sentence, str]]: A list of tuples where each tuple contains a Sentence and its associated label.
        """
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
        Generate and save cluster visualization plots based on the encoded latent representations.

        For each visualization method specified (TSNE, UMAP, PCA), it creates a corresponding plot:
             - For TSNE, a TSNEVisualizer is created, fit with the latent vectors and labels, and saved as "t_sne.png".
             - For UMAP, a UMAPVisualizer is created, fit and saved as "umap.png".
             - For PCA, a PCA visualizer from Yellowbrick is used, with labels converted to integer classes, and saved as "pca.png".

        Returns:
            A list containing the visualizer objects corresponding to each applied visualization method.
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



