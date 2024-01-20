from typing import List, Iterable, Union
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from langspace.ops.arithmetic import ArithmeticOps
from .. import LatentSpaceProbe

class ArithmeticProbe(LatentSpaceProbe):
    """
    Class for probing the arithmetic operations in the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable[Union[str, Sentence]], ops: List[ArithmeticOps]):
        """
        Initialize the ArithmeticProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (Iterable[Union[str, Sentence]]): The data to use for probing.
            ops (List[ArithmeticOps]): The arithmetic operations to evaluate.
        """
        super(ArithmeticProbe, self).__init__(model, data, 0)
        self.ops = ops

    def report(self) -> DataFrame:
        """
        Generate a report from the probe.

        Returns:
            DataFrame: The generated report.
        """
        pass
