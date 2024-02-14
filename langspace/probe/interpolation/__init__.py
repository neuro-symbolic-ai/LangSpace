from typing import List, Iterable, Union
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from langspace.metrics.interpolation import InterpolationMetrics
from .. import LatentSpaceProbe
import torch
import torch.nn.functional as F
from nltk.corpus import stopwords
from nltk import download
import gensim.downloader as api
import pandas as pd


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

    def encoding(self, data):
        """
        args: one sentence (string)
        return: latent (tensor)
        """
        seed = [data, data]
        encode_seed = self.model.decoder.tokenizer(seed, return_tensors='pt')
        encode_seed_oh = F.one_hot(encode_seed["input_ids"], num_classes=len(self.model.decoder.tokenizer.get_vocab())).to(torch.int8)
        encoded = self.model.encoder(encode_seed_oh)
        mu = encoded["embedding"][0]
        std = encoded["log_covariance"][0]
        latent, eps = self.model._sample_gauss(mu, std)
        return latent

    def decoding(self, prior):
        """
        args: tensor sent_num by latent_dim
        return: sentence list
        """
        generated = self.model.decoder(prior)['reconstruction']
        sentence_list = [s.replace(self.model.decoder.tokenizer.pad_token, "|#|")
                         for s in self.model.decoder.tokenizer.batch_decode(torch.argmax(generated, dim=-1))]
        return sentence_list

    def linearize_interpolate(self, source, target, size=10):
        return [source * (1-i/size) + target * i/size for i in range(size+1)]

    def preprocess(self, sentence, stop_words):
        return [w for w in sentence.lower().split() if w not in stop_words]

    def word_mover_distance(self, sent1, sent2, model, stopword):
        sent1 = self.preprocess(sent1, stopword)
        sent2 = self.preprocess(sent2, stopword)
        distance = model.wmdistance(sent1, sent2)
        return distance

    def interpolation_smoothness(self, interpolate_path, model_wmd, stop_words):
        """
        args: list of sentences
        return: value
        """
        source, target = interpolate_path[0], interpolate_path[-1]
        d_origin = self.word_mover_distance(source, target, model_wmd, stop_words)
        list_d = []
        for j in range(len(interpolate_path)-1):
            d = self.word_mover_distance(interpolate_path[j], interpolate_path[j+1], model_wmd, stop_words)
            list_d.append(d)

        return d_origin / sum(list_d)


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
        report = None
        for sent_pair in self.data:
            source = self.encoding(sent_pair[0])
            target = self.encoding(sent_pair[1])
            latent_path = self.linearize_interpolate(source, target)
            sent_list = self.decoding(torch.stack(latent_path))
            IS = self.interpolation_smoothness(sent_list, model_wmd, stop_words)

            source_list = [sent_pair[0] for _ in range(len(sent_list))]
            target_list = [sent_pair[1] for _ in range(len(sent_list))]
            IS_list = [IS for _ in range(len(sent_list))]

            # save to dataframe.
            d = {'source': source_list, 'target': target_list, 'distance': IS_list, 'generate': sent_list}
            res = pd.DataFrame(d)

            if report is None:
                report = res
            else:
                report.append(res)



