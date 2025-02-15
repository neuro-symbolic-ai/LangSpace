import torch
import pandas as pd
import gensim.downloader as api
from typing import Tuple, List, Dict
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk import download
from saf import Sentence
from langvae import LangVAE
from langspace.metrics.interpolation import InterpolationMetric
from .. import LatentSpaceProbe
from langspace.ops.interpolation import InterpolationOps


class InterpolationProbe(LatentSpaceProbe):
    """
    Class for probing the interpolation of the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: List[Tuple[Sentence, Sentence]],
                 eval: List[InterpolationMetric], annotations: Dict[str, List[str]] = None):
        """
        Initialize the InterpolationProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (List[Tuple[str, str]]): Sentence pairs to use for probing.
            eval (List[InterpolationMetrics]): The metrics to evaluate.
        """
        super(InterpolationProbe, self).__init__(model, data, 0)
        self.data = data
        self.eval = eval
        self.annotations = annotations

    def report(self) -> DataFrame:
        """
        Generate a report from the probe.
        Inputs:
            data = [[s1, s2], [s1, s2], ...]
            E.g., [["the appalachian mountains are a kind of mountain", "animal is a kind of living thing"]]

        Returns:
            DataFrame: The generated report.
            column = source, target, distance, generated
        """

        # calculate IS.
        model_wmd = api.load('word2vec-google-news-300')
        # model_wmd = gensim.models.KeyedVectors.load('../checkpoints/word2vec-google-news-300.model')
        download('stopwords')
        stop_words = stopwords.words('english')
        report = list()
        _, _, source, src_cvars_emb = self.encoding([sp[0] for sp in self.data], self.annotations)
        _, _, target, tgt_cvars_emb = self.encoding([sp[1] for sp in self.data], self.annotations)
        source = torch.cat([source] + src_cvars_emb, dim=-1) if src_cvars_emb else source
        target = torch.cat([target] + tgt_cvars_emb, dim=-1) if tgt_cvars_emb else target
        latent_path = torch.stack(InterpolationOps.linearize_interpolate(source, target))
        for i in range(len(self.data)):
            sent_list = self.decoding(latent_path[:, i, :])
            ismooth = InterpolationOps.interpolation_smoothness(sent_list, model_wmd, stop_words)

            # save to dataframe.
            d = {'source': self.data[i][0].surface, 'target': self.data[i][1].surface, 'distance': ismooth, 'generate': "\n".join(sent_list)}
            report.append(d)

        return pd.DataFrame(report)



