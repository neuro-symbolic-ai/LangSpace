from abc import ABC, abstractmethod
from typing import Union, Iterable
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE


class LatentSpaceProbe(ABC):
    """
    Abstract base class for probing the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable[Union[str, Sentence]], sample_size: int, **kwargs):
        """
        Initialize the LatentSpaceProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (Iterable[Union[str, Sentence]]): The data to use for probing.
            sample_size (int): The number of data points to use for probing.
            **kwargs: Additional keyword arguments.
        """
        self.model = model
        self.data = data
        self.sample_size = sample_size

    @abstractmethod
    def report(self) -> DataFrame:
        """
        Generate a report from the probe.

        Returns:
            DataFrame: The generated report.
        """
        raise NotImplementedError
