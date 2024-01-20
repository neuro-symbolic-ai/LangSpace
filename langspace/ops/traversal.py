import torch
import numpy as np
from typing import Tuple
from torch import Tensor
from scipy import stats
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

class Traverser:
    def __init__(self, dim):
        self.dim = dim

    def traverse_continuous_line(self, idx: int, size: int, loc: int = 0, scale: float = 1.0,
                                 start: float = 0.2, end: float = 0.8) -> Tensor:
        samples = np.zeros(shape=(size, self.dim))
        if idx is not None:
            cdf_traversal = np.linspace(start, end, size)
            cont_traversal = stats.norm.ppf(cdf_traversal, loc=loc, scale=scale)
            for i in range(size):
                samples[i, idx] = cont_traversal[i]
        return torch.tensor(samples.astype('f'))

    def traverse_continuous_line_control(self, idx: int, size: int, loc: int = 0, scale: float = 1.0,
                                         v: int = 0, direction: str = 'left') -> Tuple[Tensor, float]:
        samples = np.zeros(shape=(size, self.dim))
        cont_traversal = np.array([0])
        if idx is not None:
            prob = stats.norm.cdf(v, loc=loc, scale=scale)
            if direction == 'left':
                cdf_traversal = np.linspace(0.000000001, prob, size)
            else:
                cdf_traversal = np.linspace(prob, 0.999999999, size)

            cont_traversal = stats.norm.ppf(cdf_traversal, loc=loc, scale=scale)

            for i in range(size):
                samples[i, idx] = cont_traversal[i]

        return torch.tensor(samples.astype('f')), sum(cont_traversal)/len(cont_traversal)

    def traverse(self,
                 seed_sentence: str,
                 tokenizer_encoder: PreTrainedTokenizer,
                 tokenizer_decoder: PreTrainedTokenizer,
                 model: PreTrainedModel,
                 dim_z: int,
                 num_sent: int,
                 decoder:):
        seed = tokenizer_encoder.convert_tokens_to_ids(seed_sentence.split())
        encode_input = torch.tensor(seed).unsqueeze(0)
        attention_mask = (encode_input > 0).float()
        outputs = model.encoder(encode_input.long(), attention_mask)[1]
        latent_z, _, mean, logvar = model.connect_traversal(outputs)
        latent_z = latent_z.squeeze(1)
        print("Origin: ", decode_optimus(latent_z, model=model, tokenizer_decoder=tokenizer_decoder))

        for i in np.arange(dim_z, step=1):
            # randomly choose four value from normal distribution where the mean and variance from model.
            loc, scale = mean[i], math.sqrt(math.exp(logvar[i]))
            # loc, scale = 0, 1
            samples = self.traverse_continuous_line(idx=i, size=args.num_sent, loc=loc, scale=scale)
            res = decode_optimus(samples, latent_z=latent_z, model=model, tokenizer_decoder=tokenizer_decoder)

            for ix, r in enumerate(res):
                print('Dim {}, Sent {}: {}'.format(i, ix, r))
