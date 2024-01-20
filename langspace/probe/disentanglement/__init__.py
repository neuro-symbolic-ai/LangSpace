from typing import List, Iterable, Union
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from langspace.metrics import DisentanglementMetrics
from .. import LatentSpaceProbe


class DisentanglementProbe(LatentSpaceProbe):
    """
    Class for probing the disentanglement of the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable[Union[str, Sentence]], sample_size: int,
                 metrics: List[DisentanglementMetrics], gen_factors: dict):
        """
        Initialize the DisentanglementProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (Iterable[Union[str, Sentence]]): The data to use for probing.
            sample_size (int): The number of data points to use for probing.
            metrics (List[DisentanglementMetrics]): A list of disentanglement metrics to compute.
            gen_factors (dict): The generative factors to probe with.
        """
        super(DisentanglementProbe, self).__init__(model, data, sample_size)

    def report(self) -> DataFrame:
        """
        Generate a report from the probe.

        Returns:
            DataFrame: The generated report.
        """
        pass
