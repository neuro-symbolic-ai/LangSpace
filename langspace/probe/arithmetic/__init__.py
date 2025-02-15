import torch
import pandas as pd
from typing import List, Tuple, Dict
from torch import Tensor
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from langspace.ops.arithmetic import ArithmeticOps
from .. import LatentSpaceProbe


class ArithmeticProbe(LatentSpaceProbe):
    """
    Class for probing the arithmetic operations in the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: List[Tuple[Sentence, Sentence]],
                 ops: List[ArithmeticOps], annotations: Dict[str, List[str]] = None):
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
        self.annotations = annotations

    def arithmetic(self, source: Tensor, target: Tensor) -> List[Tensor]:
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
        _, _, source, src_cvars_emb = self.encoding(source_list, self.annotations)
        _, _, target, tgt_cvars_emb = self.encoding(target_list, self.annotations)
        source = torch.cat([source] + src_cvars_emb, dim=-1) if src_cvars_emb else source
        target = torch.cat([target] + tgt_cvars_emb, dim=-1) if tgt_cvars_emb else target
        latent_ops_list = self.arithmetic(source, target)
        sent_list = self.decoding(torch.cat(latent_ops_list))
        data_len = len(self.data)

        for i, op in enumerate(self.ops):
            d = {
                "source": [s.surface for s in source_list],
                "target": [s.surface for s in target_list],
                "op": [op.value.lower()] * data_len,
                "generate": sent_list[i * data_len: (i + 1) * data_len]
            }
            report.append(pd.DataFrame(d))

        return pd.concat(report).reset_index(drop=True)
