from typing import List, Iterable, Union
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from langspace.metrics.interpolation import InterpolationMetrics
from .. import LatentSpaceProbe


class InterpolationProbe(LatentSpaceProbe):
    """
    Class for probing the interpolation of the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable[Union[str, Sentence]], eval: List[InterpolationMetrics]):
        """
        Initialize the InterpolationProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (Iterable[Union[str, Sentence]]): The data to use for probing.
            eval (List[InterpolationMetrics]): The metrics to evaluate.
        """
        super(InterpolationProbe, self).__init__(model, data, 0)
        self.eval = eval

    def report(self) -> DataFrame:
        """
        Generate a report from the probe.

        Returns:
            DataFrame: The generated report.
        """
        pass
