import torch
import numpy as np
from typing import Tuple
from scipy import stats
from torch import Tensor


class TraversalOps:
    @staticmethod
    def dimension_random_walk(mu: Tensor, logvar: Tensor, latent: Tensor, dim: int, sample_size: int) -> Tensor:
        """
        args:
            mu = [...], logvar = [...], latent type: list
            dim: target traversal dimension
        return:
            [ [], [], ..., [] ]
        """
        sample_list = latent.repeat(1, sample_size).view(latent.shape[0], sample_size, latent.shape[1])
        loc, scale = mu[:, dim], torch.sqrt(torch.exp(logvar[:, dim]))
        cdf_traversal = np.linspace(0.001, 0.999, sample_size)
        cont_traversal = torch.stack(
            [torch.tensor(stats.norm.ppf(cdf_traversal, loc=loc[i].item(), scale=scale[i].item()))
             for i in range(latent.shape[0])]
        )  # sample list for dim i
        sample_list[:, :, dim] = cont_traversal
        return sample_list

    @staticmethod
    def calculate_distance(seed: Tensor, samples: Tensor) -> Tensor:
        """
        Compute the length (norm) of the distance between the vectors
        args: seed, sample (list)
        return: distance list
        """
        d = samples - seed.repeat(1, samples.shape[1]).view(seed.shape[0], samples.shape[1], seed.shape[1])
        res = torch.sqrt(torch.einsum('bij,bij->bi', d, d))
        return res

    @staticmethod
    def traverse(mu: Tensor, std: Tensor, latent: Tensor, dim: int, sample_size: int) -> Tuple[Tensor, Tensor]:
        # traversal
        prior_latent = TraversalOps.dimension_random_walk(mu, std, latent, dim, sample_size)
        # distance
        dist = TraversalOps.calculate_distance(latent, prior_latent)

        return prior_latent, dist
