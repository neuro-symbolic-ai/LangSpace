from typing import List, Iterable, Union
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from .. import LatentSpaceProbe


class TraversalProbe(LatentSpaceProbe):
    """
    Class for probing the traversal of the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable[Union[str, Sentence]], sample_size: int, dims: List[int]):
        """
        Initialize the TraversalProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (Iterable[Union[str, Sentence]]): The data to use for probing.
            sample_size (int): The number of data points to use for probing.
            dims (List[int]): The dimensions to traverse.
        """
        super(TraversalProbe, self).__init__(model, data, sample_size)
        self.dims = dims

    def report(self) -> DataFrame:
        """
        Generate a report from the probe.

        Returns:
            DataFrame: The generated report.
        """
        pass

