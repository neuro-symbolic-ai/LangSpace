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
    A probe for evaluating interpolation in the latent space of an LM-VAE.

    This class facilitates the exploration of latent space by interpolating between pairs of sentence encodings.

    Attributes:
        model (LangVAE): The language VAE model to be probed.
        data (List[Tuple[Sentence, Sentence]]): A list of sentence pairs (source and target) for interpolation.
        eval (List[InterpolationMetric]): A list of evaluation metrics used to assess the interpolation quality.
        annotations (Dict[str, List[str]]): Optional dictionary of annotation types to be processed and all their
        possible values, for conditional encoding.
    """
    def __init__(self, model: LangVAE, data: List[Tuple[Sentence, Sentence]],
                 eval: List[InterpolationMetric], annotations: Dict[str, List[str]] = None):
        """
        Initialize the InterpolationProbe with a specified model, sentence pairs, evaluation metrics, and optional annotations.

        Args:
            model (LangVAE): The language VAE model to be probed.
            data (List[Tuple[Sentence, Sentence]]): A list of sentence pairs. Each tuple represents a source and a target Sentence.
            eval (List[InterpolationMetric]): A list of interpolation metrics to evaluate the smoothness or quality of the interpolation.
            annotations (Dict[str, List[str]], optional): Optional dictionary of annotation types to be processed and
            all their possible values, for conditional encoding.
        """
        super(InterpolationProbe, self).__init__(model, data, 0)
        self.data = data
        self.eval = eval
        self.annotations = annotations

    def report(self) -> DataFrame:
        """
        Generate a detailed report of the interpolation results across the latent space.

        Returns:
            DataFrame: A report containing the interpolation results. Each row corresponds to a sentence pair with:
                     - 'source': Surface text of the source sentence.
                     - 'target': Surface text of the target sentence.
                     - 'distance': The computed smoothness (or distance) metric.
                     - 'generate': A newline-separated string of the generated interpolation sentences.
        """

        # calculate IS.
        model_wmd = api.load('word2vec-google-news-300')
        # model_wmd = gensim.models.KeyedVectors.load('../checkpoints/word2vec-google-news-300.model')
        download('stopwords')
        stop_words = stopwords.words('english')
        report = list()
        _, _, source, src_cvars_emb = self.encoding([sp[0] for sp in self.data], self.annotations)
        _, _, target, tgt_cvars_emb = self.encoding([sp[1] for sp in self.data], self.annotations)
        source = torch.cat([source] + src_cvars_emb, dim=-1) if (src_cvars_emb and self.model.decoder.conditional) else source
        target = torch.cat([target] + tgt_cvars_emb, dim=-1) if (tgt_cvars_emb and self.model.decoder.conditional) else target
        latent_path = torch.stack(InterpolationOps.linearize_interpolate(source, target))
        for i in range(len(self.data)):
            sent_list = self.decoding(latent_path[:, i, :])
            ismooth = InterpolationOps.interpolation_smoothness(sent_list, model_wmd, stop_words)

            # save to dataframe.
            d = {'source': self.data[i][0].surface, 'target': self.data[i][1].surface, 'distance': ismooth, 'generate': "\n".join(sent_list)}
            report.append(d)

        return pd.DataFrame(report)



