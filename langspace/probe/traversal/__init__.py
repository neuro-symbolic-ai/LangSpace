from typing import List, Iterable, Union
import pandas as pd
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from .. import LatentSpaceProbe
import math
from langspace.ops.traversal import *
import torch.nn.functional as F


class TraversalProbe(LatentSpaceProbe):
    """
    Class for probing the traversal of the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable[Union[str, Sentence]], sample_size: int, dims: List[int]):
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

    def encoding(self, data):
        """
        args: single sentence (string)
        return, mu, sigma, latent (list)
        """
        seed = [data, data]
        encode_seed = self.model.decoder.tokenizer(seed, return_tensors='pt')
        encode_seed_oh = F.one_hot(encode_seed["input_ids"], num_classes=len(self.model.decoder.tokenizer.get_vocab())).to(torch.int8)
        encoded = self.model.encoder(encode_seed_oh)
        mu = encoded["embedding"][0]
        std = encoded["log_covariance"][0]
        latent, eps = self.model._sample_gauss(mu, std)
        return mu.tolist(), std.tolist(), latent.tolist()

    def decoding(self, prior):
        """
        args: sent_num by latent_dim
        return: sentence list
        """
        generated = self.model.decoder(prior)['reconstruction']
        sentence_list = [s.replace(self.model.decoder.tokenizer.pad_token, "|#|")
                         for s in self.model.decoder.tokenizer.batch_decode(torch.argmax(generated, dim=-1))]
        return sentence_list

    def dimension_random_walk(self, mu, logvar, latent, dim):
        """
        args:
            mu = [...], logvar = [...], latent type: list
            dim: target traversal dimension
        return:
            [ [], [], ..., [] ]
        """
        sample_list = np.array([latent for _ in range(self.sample_size)])
        loc, scale = mu[dim], math.sqrt(math.exp(logvar[dim]))
        cdf_traversal = np.linspace(0.001, 0.999, self.sample_size)
        cont_traversal = stats.norm.ppf(cdf_traversal, loc=loc, scale=scale) # sample list for dim i
        sample_list[:, dim] = cont_traversal
        return sample_list

    def calculate_distance(self, seed, samples):
        """
        Compute the length (norm) of the distance between the vectors
        args: seed, sample (list)
        return: distance list
        """
        d = np.subtract(np.array(samples), np.array(seed))
        res = np.sqrt(np.einsum('ij,ij->i',d,d))
        return res

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
        report = None
        for sent in self.data:
            # encoding
            mu, std, latent = self.encoding(sent)
            for dim in self.dims:
                # traversal
                prior_latent = torch.tensor(self.dimension_random_walk(mu, std, latent, dim)).float()
                # distance
                dist = self.calculate_distance(latent, prior_latent)
                # decoding
                sent_list = self.decoding(prior_latent)
                # Returns a pandas.DataFrame with column {'seeds', 'dim', 'distance', 'generate'}
                c1 = [sent for _ in range(len(sent_list))]
                c2 = dist
                c3 = sent_list

                # save to dataframe.
                d = {'seeds': c1, 'dim': dim, 'distance': c2, 'generate': c3}
                res = pd.DataFrame(d)

                if report is None:
                    report = res
                else:
                    report.append(res)

        return report










