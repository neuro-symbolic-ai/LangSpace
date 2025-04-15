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
    A probe for exploring arithmetic operations in the latent space of a language model variational autoencoder (LM-VAE).

    This class applies specified arithmetic operations to latent representations obtained from pairs
    of sentences. It supports operations such as summation, subtraction, and averaging. In addition,
    the probe can generate a report in the form of a pandas DataFrame summarizing the original source
    and target sentences alongside the results of the applied operations.

    Attributes:
        model (LangVAE): The LM-VAE model to be probed.
        data (List[Tuple[Sentence, Sentence]]): A list of sentence pairs as (source, target) tuples.
        ops (List[ArithmeticOps]): A list of arithmetic operations to be applied to the latent vectors.
        annotations (Dict[str, List[str]], optional): Dictionary of annotation types to be processed and all their
        possible values.
    """
    def __init__(self, model: LangVAE, data: List[Tuple[Sentence, Sentence]],
                 ops: List[ArithmeticOps], annotations: Dict[str, List[str]] = None):
        """
        Initialize the ArithmeticProbe with a LM-VAE model, data pairs, arithmetic operations,
        and optional annotations.

        Args:
            model (LangVAE): The LM-VAE model whose latent space is to be probed.
            data (List[Tuple[Sentence, Sentence]]): A list of tuples, each containing a source and target Sentence.
            ops (List[ArithmeticOps]): A list of arithmetic operations (e.g., SUM, SUB, AVG) to perform.
            annotations (Dict[str, List[str]], optional): Optional dictionary of annotation types to be processed and
            all their possible values, for conditional encoding.
        """
        super(ArithmeticProbe, self).__init__(model, data, 0)
        self.data = data
        self.ops = ops
        self.annotations = annotations

    def arithmetic(self, source: Tensor, target: Tensor) -> List[Tensor]:
        """
        Apply arithmetic operations to the source and target latent representations.

        Args:
            source (Tensor): The latent representation of the source sentences.
            target (Tensor): The latent representation of the target sentences.

        Returns:
            A list of tensors, each resulting from applying the corresponding arithmetic operation.
        """
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
        Generate a report summarizing the arithmetic probe results.

        The final DataFrame will have the following columns:
            - `source`: The original source sentence surfaces.
            - `target`: The original target sentence surfaces.
            - `op`: The arithmetic operation applied (as a lowercase string).
            - `generate`: The generated sentence after applying the latent operation.

        Returns:
            A DataFrame containing a detailed report for each arithmetic operation.
        """
        report = list()
        source_list = [sp[0] for sp in self.data]
        target_list = [sp[1] for sp in self.data]
        _, _, source, src_cvars_emb = self.encoding(source_list, self.annotations)
        _, _, target, tgt_cvars_emb = self.encoding(target_list, self.annotations)
        source = torch.cat([source] + src_cvars_emb, dim=-1) if (src_cvars_emb and self.model.decoder.conditional) else source
        target = torch.cat([target] + tgt_cvars_emb, dim=-1) if (tgt_cvars_emb and self.model.decoder.conditional) else target
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
