from typing import List, Iterable, Union
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from .. import LatentSpaceProbe


class STSProbe(LatentSpaceProbe):
    """
    Class for probing the semantic textual similarity in the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable[Union[str, Sentence]], sample_size: int):
        """
        Initialize the STSProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (Iterable[Union[str, Sentence]]): The data to use for probing.
            sample_size (int): The number of data points to use for probing.
        """
        super(STSProbe, self).__init__(model, data, sample_size)

    def report(self) -> DataFrame:
        """
        Generate a report from the probe.

        Returns:
            DataFrame: The generated report.
        """
        pass
