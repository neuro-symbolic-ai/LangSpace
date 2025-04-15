import torch
import numpy as np
from typing import Tuple
from scipy import stats
from torch import Tensor


class TraversalOps:
    """
    Class for performing latent space traversal and distance computations.

    This class provides methods that enable controlled exploration of generative latent spaces.


    """
    @staticmethod
    def dimension_random_walk(mu: Tensor, logvar: Tensor, latent: Tensor, dim: int, sample_size: int) -> Tensor:
        """
        Performs a random walk along a specified latent dimension.

        This method generates a sequence of latent vectors by replacing the values at a target dimension
        with samples drawn from the inverse cumulative density function (percent point function) of the normal
        distribution parameterized by the corresponding mean and log variance. The sampled values span a
        continuum of percentiles from 0.001 to 0.999, thereby exploring the latent space along that specific axis.

        The controlled perturbation along one dimension helps reveal how changes in a single latent feature influence
        the generated output, a key aspect in the analysis of disentangled representations.

        Args:
            mu (Tensor): Mean tensor for the latent distribution (shape: [batch_size, latent_dim]).
            logvar (Tensor): Log variance tensor for the latent distribution (shape: [batch_size, latent_dim]).
            latent (Tensor): Original latent vector tensor (shape: [batch_size, latent_dim]).
            dim (int): Target latent dimension index along which to perform the traversal.
            sample_size (int): Number of samples (or steps) to generate along the traversal.

        Returns:
            A tensor containing traversed latent vectors with shape [batch_size, sample_size, latent_dim].
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
        Computes the Euclidean distance between a seed latent vector and a set of sample latent vectors.

        This method calculates the L2 norm, i.e., the square root of the sum of squared differences, for each
        sample in relation to the original seed vector.

        Args:
            seed (Tensor): The original latent vector tensor (shape: [batch_size, latent_dim]).
            samples (Tensor): A tensor of sample latent vectors (shape: [batch_size, sample_size, latent_dim]).

        Returns:
            A tensor containing the Euclidean distances (L2 norms) with shape [batch_size, sample_size].
        """
        d = samples - seed.repeat(1, samples.shape[1]).view(seed.shape[0], samples.shape[1], seed.shape[1])
        res = torch.sqrt(torch.einsum('bij,bij->bi', d, d))
        return res

    @staticmethod
    def traverse(mu: Tensor, std: Tensor, latent: Tensor, dim: int, sample_size: int) -> Tuple[Tensor, Tensor]:
        """
        Integrates latent space traversal and distance computation on a specified latent dimension.

        This high-level method performs two main operations:
          1. It generates a series of latent vectors by executing a random walk along the given dimension,
             using the provided mean and (log-)variance parameters.
          2. It computes the Euclidean distances between the original latent vector and each of the traversed samples,
             quantifying the extent of perturbation.

        Args:
            mu (Tensor): Mean tensor for the latent distribution (shape: [batch_size, latent_dim]).
            std (Tensor): Standard deviation or log variance used as a surrogate for standard deviation (shape: [batch_size, latent_dim]).
            latent (Tensor): Original latent vector tensor (shape: [batch_size, latent_dim]).
            dim (int): Target latent dimension to manipulate.
            sample_size (int): Number of traversal samples to generate.

        Returns:
            Tuple[Tensor, Tensor]:
                - A tensor containing the traversed latent vectors (shape: [batch_size, sample_size, latent_dim]).
                - A tensor with the Euclidean distances (L2 norms) of each traversal from the original latent vector
                  (shape: [batch_size, sample_size]).
        """
        # traversal
        prior_latent = TraversalOps.dimension_random_walk(mu, std, latent, dim, sample_size)
        # distance
        dist = TraversalOps.calculate_distance(latent, prior_latent)

        return prior_latent, dist
