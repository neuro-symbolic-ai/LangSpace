import torch
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Iterable
from torch import Tensor
from pandas import DataFrame
from tqdm import tqdm
from saf import Sentence
from saf_datasets import SentenceDataSet, BasicSentenceDataSet
from langvae import LangVAE
from langvae.data_conversion.tokenization import TokenizedDataSet, TokenizedAnnotatedDataSet


class LatentSpaceProbe(ABC):
    """
    Abstract base class for probing the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable[Sentence], sample_size: int, **kwargs):
        """
        Initialize the LatentSpaceProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (Iterable[Sentence]): The data to use for probing.
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

    def get_tokenized_data_seed(self, data: Iterable[Sentence],
                                annotations: Dict[str, List[str]] = None) -> TokenizedDataSet:
        if (not isinstance(data, SentenceDataSet)):
            data = BasicSentenceDataSet(list(data))

        if (annotations):
            seed = TokenizedAnnotatedDataSet(data,
                                             self.model.decoder.tokenizer,
                                             self.model.decoder.max_len,
                                             annotations=annotations)
        else:
            seed = TokenizedDataSet(data, self.model.decoder.tokenizer, self.model.decoder.max_len)

        return seed

    def encoding(self, data: Iterable[Sentence],
                 annotations: Dict[str, List[str]] = None) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        """
        Encode the input data and return the mean, standard deviation, and latent representation.

        Args:
            data (Iterable[Union[str, Sentence]]): The input data to encode.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: A tuple containing the mean, standard deviation,
            latent representation and conditional variable embeddings, as tensors.
        """
        seed = self.get_tokenized_data_seed(data, annotations)

        with torch.no_grad():
            cond_vars = {annot: seed[:][annot] for annot in annotations} if (annotations) else None
            encoded = self.model.encoder(seed[:]["data"], cond_vars)
            mu, log_var = encoded.embedding, encoded.log_covariance
            cvars_emb = encoded.cvars_embedding
            std = torch.exp(0.5 * log_var)
            z, eps = self.model._sample_gauss(mu, std)

        return mu, std, z, cvars_emb

    def batched_encoding(self, data: Iterable[Sentence], annotations: Dict[str, List[str]] = None,
                         batch_size: int = 100) -> Tensor:
        """
        Encodes the sentences

        Args:
            data (Iterable[Sentence]): sentences
            annotations (List[str]): optional annotations to be used, if available in the data, and their respective
            possible values.
            batch_size (int): number of sentences to be processed simultaneously

        Returns:
            Tensor: Latent representation
        """
        seed = self.get_tokenized_data_seed(data, annotations)

        latent = list()
        for i in tqdm(range(len(seed) // batch_size + int(len(seed) % batch_size > 0)), desc="Encoding"):
            encode_seed = seed[i * batch_size: i * batch_size + batch_size]
            cond_vars = {annot: encode_seed[annot] for annot in annotations} if (annotations) else None

            with torch.no_grad():
                z, cvars_emb = self.model.encode_z(encode_seed["data"], cond_vars)

            latent.append(z)

        return torch.cat(latent)

    def decoding(self, prior: Tensor, cvars_emb: List[Tensor] = None) -> List[str]:
        """
        args: sent_num by latent_dim
        return: sentence list
        """
        sentence_list = self.model.decode_sentences(prior, cvars_emb)

        return sentence_list
