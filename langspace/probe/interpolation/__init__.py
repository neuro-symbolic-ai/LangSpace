import torch
import torch.nn.functional as F
import pandas as pd
import gensim.downloader as api
from typing import Tuple, List
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
    def __init__(self, model: LangVAE, data: List[Tuple[str, str]], eval: List[InterpolationMetric]):
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

    def encoding(self, data: List[str]):
        """
        Encodes the sentences

        Args:
            data (List[str]): sentences

        Returns:
            Tensor: Latent representation
        """
        seed = list(data)
        if (len(seed) < 2):
            seed.append("")

        encode_seed = self.model.decoder.tokenizer(seed, padding="max_length", truncation=True,
                                                   max_length=self.model.decoder.max_len, return_tensors='pt')
        encode_seed_oh = F.one_hot(encode_seed["input_ids"], num_classes=len(self.model.decoder.tokenizer.get_vocab())).to(torch.int8)
        with torch.no_grad():
            latent = self.model.encode_z(encode_seed_oh)

        return latent

    def decoding(self, prior):
        """
        Decodes latent representations

        Args:
            prior (Tensor): latent representations

        Returns:
            List[str]: Decoded sentences
        """

        generated = self.model.decoder(prior)['reconstruction']
        sentence_list = self.model.decoder.tokenizer.batch_decode(torch.argmax(generated, dim=-1),
                                                                  skip_special_tokens=True)
        return sentence_list

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
        source = self.encoding([sp[0] for sp in self.data])
        target = self.encoding([sp[1] for sp in self.data])
        latent_path = torch.stack(InterpolationOps.linearize_interpolate(source, target))
        for i in range(len(self.data)):
            sent_list = self.decoding(latent_path[:, i, :])
            ismooth = InterpolationOps.interpolation_smoothness(sent_list, model_wmd, stop_words)

            # save to dataframe.
            d = {'source': self.data[i][0], 'target': self.data[i][1], 'distance': ismooth, 'generate': "\n".join(sent_list)}
            report.append(d)

        return pd.DataFrame(report)



