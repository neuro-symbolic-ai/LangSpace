from typing import List
from huggingface_hub import list_models


def select_models(encoder: str = None,
                  decoder: str = None,
                  latent_dim: int = 0,
                  annotations: List[str] = None,
                  conditional: bool = None) -> List[str]:
    """
    Selects a list of LM-VAE models available from the neuro-symbolic-ai repository, according to the specified criteria.

    :param encoder: The name of the encoder model (e.g., bert-base-cased, flan-t5-base)
    :param decoder: The name of the decoder model (e.g., gpt2, Llama-3.2-3B)
    :param latent_dim: The latent dimension of the LM-VAE (e.g., 64, 128)
    :param annotations: Annotations the model was trained on (e.g., srl)
    :param conditional: If it is a conditional variable model
    :return: A list of available modes
    """
    models = [model_info.id.lower() for model_info in list_models(author="neuro-symbolic-ai")]
    encoder = encoder.lower()
    decoder = decoder.lower()

    filtered = [
        model_name for model_name in models
        if (
            (encoder is None or encoder in model_name) and
            (decoder is None or decoder in model_name) and
            (latent_dim == 0 or f"_l{latent_dim}" in model_name) and
            (annotations is None or f"_{'-'.join([annot.lower() for annot in annotations])}" in model_name) and
            (conditional is None or (conditional and "langcvae" in model_name) or (not conditional and "langvae" in model_name))
        )
    ]

    return filtered
