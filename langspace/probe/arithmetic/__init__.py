import torch.nn.functional as F
import torch
import pandas as pd
from typing import List, Tuple
from torch import Tensor
from pandas import DataFrame
from langvae import LangVAE
from langspace.ops.arithmetic import ArithmeticOps
from .. import LatentSpaceProbe


class ArithmeticProbe(LatentSpaceProbe):
    """
    Class for probing the arithmetic operations in the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: List[Tuple[str, str]], ops: List[ArithmeticOps]):
        """
        Initialize the ArithmeticProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (Iterable[Union[str, Sentence]]): The data to use for probing.
            ops (List[ArithmeticOps]): The arithmetic operations to evaluate.
        """
        super(ArithmeticProbe, self).__init__(model, data, 0)
        self.data = data
        self.ops = ops

    def encoding(self, data):
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

    def arithmetic(self, source, target) -> List[Tensor]:
        res = list()
        for op in self.ops:
            if op == ArithmeticOps.SUM:
                res.append(source + target)
            elif op == ArithmeticOps.SUB:
                res.append(source - target)
            else:
                res.append((source + target) / 2)

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
        report = list()
        source_list = [sp[0] for sp in self.data]
        target_list = [sp[1] for sp in self.data]
        source = self.encoding(source_list)
        target = self.encoding(target_list)
        latent_ops_list = self.arithmetic(source, target)
        sent_list = self.decoding(torch.cat(latent_ops_list))
        data_len = len(self.data)

        for i, op in enumerate(self.ops):
            d = {
                "source": source_list,
                "target": target_list,
                "op": [op.value.lower()] * data_len,
                "generate": sent_list[i * data_len: (i + 1) * data_len]
            }
            report.append(pd.DataFrame(d))

        return pd.concat(report).reset_index(drop=True)
