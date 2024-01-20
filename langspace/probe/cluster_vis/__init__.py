from typing import List, Iterable, Union
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from .. import LatentSpaceProbe
from .methods import ClusterVisualizationMethod


class ClusterVisualizationProbe(LatentSpaceProbe):
    def __init__(self, model: LangVAE, data: Iterable[Union[str, Sentence]], sample_size: int,
                 method: List[ClusterVisualizationMethod]):
        super(ClusterVisualizationProbe, self).__init__(model, data, sample_size)

    def report(self) -> DataFrame:
        pass
