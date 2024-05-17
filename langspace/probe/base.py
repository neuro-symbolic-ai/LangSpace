import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Union, Iterable
from pandas import DataFrame
from tqdm import tqdm
from saf import Sentence
from langvae import LangVAE


class LatentSpaceProbe(ABC):
    """
    Abstract base class for probing the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable, sample_size: int, **kwargs):
        """
        Initialize the LatentSpaceProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (Iterable[Union[str, Sentence]]): The data to use for probing.
            sample_size (int): The number of data points to use for probing.
            **kwargs: Additional keyword arguments.
        """
        self.model = model
        self.data = data
        self.sample_size = sample_size

    @abstractmethod
    def report(self) -> DataFrame:
        """
        Generate a report from the probe.

        Returns:
            DataFrame: The generated report.
        """
        raise NotImplementedError

    def encoding(self, data, batch_size: int = 100):
        """
        Encodes the sentences

        Args:
            data (List[str]): sentences
            batch_size (int): number of sentences to be processed simultaneously

        Returns:
            Tensor: Latent representation
        """
        seed = list(data)
        if (len(seed) < 2):
            seed.append("")

        latent = list()
        for i in tqdm(range(len(seed) // batch_size + int(len(seed) % batch_size > 0)), desc="Encoding"):
            encode_seed = self.model.decoder.tokenizer(seed[i * batch_size: i * batch_size + batch_size],
                                                       padding="max_length", truncation=True,
                                                       max_length=self.model.decoder.max_len,
                                                       return_tensors='pt')
            encode_seed_oh = F.one_hot(encode_seed["input_ids"],
                                       num_classes=len(self.model.decoder.tokenizer.get_vocab())).to(torch.int8)
            with torch.no_grad():
                z = self.model.encode_z(encode_seed_oh)
            latent.append(z)

        return torch.cat(latent)
