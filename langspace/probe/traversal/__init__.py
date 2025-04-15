from typing import Tuple, List, Dict, Iterable, Union
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from pandas import DataFrame
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from saf import Sentence
from saf_datasets import SentenceDataSet, BasicSentenceDataSet
from langvae import LangVAE
from langvae.data_conversion.tokenization import TokenizedDataSet, TokenizedAnnotatedDataSet
from .. import LatentSpaceProbe
from langspace.ops.traversal import TraversalOps


class TraversalProbe(LatentSpaceProbe):
    """
    A probe for analyzing latent space traversal in an LM-VAE.

    This probe performs a systematic traversal along specified latent dimensions. It first encodes a set
    of input sentences into latent representations, then perturbs these representations along each of
    the desired dimensions. The perturbed latent points are subsequently decoded back into sentences,
    allowing the user to inspect how modifications in particular latent dimensions affect the generated output.

    Attributes:
        dims (List[int]): List of latent space dimensions along which the probe will traverse.
        model (LangVAE): The LM-VAE model to be probed.
        sample_size (int): The number of samples generated along each dimension.
        annotations (Dict[str, List[str]], optional): Optional dictionary of annotation types to be processed and all
        their possible values, for conditional encoding.
    """

    def __init__(self, model: LangVAE, data: Iterable[Sentence], sample_size: int, dims: List[int],
                 annotations: Dict[str, List[str]] = None):
        """
        Initialize the TraversalProbe with a specified model, dataset, and traversal configurations.

        Args:
            model (LangVAE): The language VAE model to probe.
            data (Iterable[Sentence]): An iterable of sentences to be encoded.
            sample_size (int): The number of traversal samples to generate for each specified dimension.
            dims (List[int]): A list of indices representing the latent space dimensions to traverse.
            annotations (Dict[str, List[str]], optional): Optional dictionary of annotation types to be processed and all
            their possible values, for conditional encoding.
        """
        super(TraversalProbe, self).__init__(model, data, sample_size)
        self.dims = dims
        self.model = model
        self.sample_size = sample_size
        self.annotations = annotations

    def report(self) -> DataFrame:
        """
        Generate a report detailing the latent space traversal results.

        Returns:
            DataFrame: A pandas DataFrame containing the traversal report with the following columns:
                - 'seeds': The source sentences.
                - 'dim': The latent dimension traversed.
                - 'distance': The perturbation magnitude used in the traversal.
                - 'generate': The generated sentence from the traversed latent point.
        """
        report = list()
        # encoding
        print("Encoding...")
        mu, std, latent, cvars_emb = self.encoding(self.data, self.annotations)
        latent = torch.cat([latent] + cvars_emb, dim=-1) if (cvars_emb and self.model.decoder.conditional) else latent
        print("Traversing...")
        with Parallel(n_jobs=cpu_count(True)) as ppool:
            prior_latents_dists = ppool(delayed(TraversalOps.traverse)(mu, std, latent, dim, self.sample_size)
                                        for dim in self.dims)

        for idx, dim in tqdm(enumerate(self.dims), desc="Decoding dim"):
            prior_latent, dist = prior_latents_dists[idx]
            # decoding
            pl_dims = prior_latent.shape
            sent_lists = self.decoding(prior_latent.view(pl_dims[0] * pl_dims[1], pl_dims[2]))

            for j, gen_sent in enumerate(sent_lists):
                i = j // self.sample_size
                sent = self.data[i]
                if isinstance(sent, Sentence):
                    sent = sent.surface
                d = {'seeds': sent, 'dim': dim, 'distance': dist[i][j % self.sample_size].item(), 'generate': gen_sent}
                # save to dataframe.
                report.append(d)

            report.sort(key=lambda x: (x["seeds"], x["dim"], x["distance"]))

        return pd.DataFrame(report)










