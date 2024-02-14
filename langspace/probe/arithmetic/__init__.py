from typing import List, Iterable, Union
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from langspace.ops.arithmetic import ArithmeticOps
from .. import LatentSpaceProbe
import torch.nn.functional as F
import torch
import pandas as pd


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

    def arithmetic(self, source, target):
        res = []
        for ops in self.ops:
            if ops == 'add':
                res.append(source + target)
            elif ops == 'sub':
                res.append(source - target)
            else:
                res.append((source + target)/2)
        return res

    def report(self) -> DataFrame:
        """
        Generate a report from the probe.
        Input:
            data = [[s1, s2], [s1, s2], ...]
            E.g., [["the appalachian mountains are a kind of mountain", "animal is a kind of living thing"]]

            ops = ['add', 'sub', 'avg']

        Returns:
            DataFrame: The generated report.
            column = source, target, ops, generated.
        """
        report = None
        for sent_pair in self.data:
            source = self.encoding(sent_pair[0])
            target = self.encoding(sent_pair[1])
            latent_ops_list = self.arithmetic(source, target)
            sent_list = self.decoding(torch.stack(latent_ops_list))
            # save to dataframe.
            source_list = [sent_pair[0] for _ in range(len(sent_list))]
            target_list = [sent_pair[1] for _ in range(len(sent_list))]
            d = {'source': source_list, 'target': target_list, 'ops': self.ops, 'generate': latent_ops_list}
            res = pd.DataFrame(d)

            if report is None:
                report = res
            else:
                report.append(res)
