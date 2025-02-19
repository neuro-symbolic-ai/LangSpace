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
    Class for probing the traversal of the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable[Sentence], sample_size: int, dims: List[int],
                 annotations: Dict[str, List[str]] = None):
        """
        Initialize the TraversalProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (Iterable[Union[str, Sentence]]): The data to use for probing. "plain sentence"
            sample_size (int): The number of data points to use for probing.
            dims (List[int]): The dimensions to traverse.
        """
        super(TraversalProbe, self).__init__(model, data, sample_size)
        self.dims = dims
        self.model = model
        self.sample_size = sample_size
        self.annotations = annotations

    def report(self) -> DataFrame:
        """
        Generate a report from the probe.
        Inputs:
            data = [s1, s2, ...]
            E.g., ["the appalachian mountains are a kind of mountain", "the appalachian mountains are a kind of mountain"]

            sample_size = 10 number of sample for each dimension

            dim = [8, 10, 1, ...] dimension to traverse

        Returns:
            DataFrame: The generated report.
            column: d = {'seeds' , 'dim', 'distance', 'generate'}
        """
        report = list()
        # encoding
        print("Encoding...")
        mu, std, latent, cvars_emb = self.encoding(self.data, self.annotations)
        latent = torch.cat([latent] + cvars_emb, dim=-1) if (cvars_emb and self.model.decoder.conditional) else latent
        print("Traversing...")
        with Parallel(n_jobs=1) as ppool:
            prior_latents_dists = ppool(delayed(TraversalOps.traverse)(mu, std, latent, dim, self.sample_size)
                                        for dim in self.dims)

        for dim in tqdm(self.dims, desc="Decoding dim"):
            prior_latent, dist = prior_latents_dists[dim]
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










