from typing import List, Iterable, Union
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from langspace.metrics import DisentanglementMetrics
from .. import LatentSpaceProbe


class DisentanglementProbe(LatentSpaceProbe):
    def __init__(self, model: LangVAE, data: Iterable[Union[str, Sentence]], sample_size: int,
                 metrics: List[DisentanglementMetrics], gen_factors: dict):
        super(DisentanglementProbe, self).__init__(model, data, sample_size)

    def report(self) -> DataFrame:
        pass
