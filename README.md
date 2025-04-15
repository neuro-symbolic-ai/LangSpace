# LangSpace: Probing Large Language VAEs made simple

LangSpace is a Python library for evaluating and probing language models Variational Autoencoders (LM-VAEs). It provides an easy-to-use interface to perform a variety of analises on pretrained [LangVAE](https://github.com/neuro-symbolic-ai/LangVAE) models.

## Why LangSpace?

While there are a variety of tools and benchmarks dedicated to the evaluation of text embeddings, LangSpace aims to be a comprehensive resource focused on the analysis of LM-VAE latent spaces. LM-VAEs can encode the knowledge of their pre-trained components into compact latent vectors and enables guided language generation from an abstract level using said vectors. The benefits of such models also extend to interpretability (due to their better disentanglement properties), as the VAE architectural bottleneck provides a singular point for probing a model’s latent space structure and its syntactic/semantic representation and inferential properties.

As a companion framework to [LangVAE](https://github.com/neuro-symbolic-ai/LangVAE), LangSpace provides a simple API to perform a variety of analyses on pre-trained LM-VAEs models, namely:

 - Probes: vector arithmetic and interpolation, latent space traversal, disentanglement and cluster
visualisation. 
 - Metrics: disentanglement (z-diff, z-min-var, MIG,
Disentanglement, Informativeness, Completeness),
interpolation (quality, smoothness).

## Installation

To install LangSpace, simply run:

```bash
pip install langspace
```

This will install all necessary dependencies and set up the package for use in your Python projects.

## Quick start

Here's a basic example of how to perform a disentanglement evaluation and an interpolation probe on an LM-VAE model trained with LangVAE:

```python
import torch
import nltk
from langvae import LangVAE
from saf_datasets import EntailmentBankDataSet
from langspace.probe import DisentanglementProbe
from langspace.metrics.disentanglement import DisentanglementMetric as Metric
from langspace.probe import InterpolationProbe
from langspace.metrics.interpolation import InterpolationMetric as InterpMetric
from saf.importers import ListImporter

# Load annotated data from saf_datasets.
dataset = EntailmentBankDataSet.from_resource("pos+lemma+ctag+dep+srl#expl_only-noreps")
annotations = {"srl_f": dataset.annotations["srl"]}

# The 'srl' annotation contains a list with the role of a single token in each phrase in the sentence.
# 'srl_f' will contain the first non-empty srl annotation for each token.
for sent in dataset:
    for token in sent.tokens:
        srl = token.annotations["srl"]
        token_annot = [lbl for lbl in srl if (lbl != "O")][0] if (len(set(srl)) > 1) else srl[0]
        token.annotations["srl_f"] = token_annot
        

# Load explanation LM-VAE for generation.
model = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langcvae-bert-base-cased-gpt2-srl-l128") # Loads model from HuggingFace Hub.
model.eval()

if (torch.cuda.is_available()):
  model.encoder.to("cuda")
  model.decoder.to("cuda")
  model.encoder.init_pretrained_model()
  model.decoder.init_pretrained_model()


# Probing latent disentanglement
gen_factors = {
    "direction": ["ARGM-DIR"],
    "because": ["ARGM-CAU"],
    "purpose": ["ARGM-PRP","ARGM-PNC", "ARGM-GOL"],
    "more": ["ARGM-EXT"],
    "location": ["ARGM-LOC"],
    "argument": ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4"],
    "manner": ["ARGM-MNR"],
    "can": ["ARGM-MOD"],
    "argm-prd": ["ARGM-PRD"],
    "empty": ["O"],
    "negation": ["ARGM-NEG"],
    "verb": ["V"],
    "if-then": ["ARGM-ADV", "ARGM-DIS"],
    "time": ["ARGM-TMP"],
    "C-ARG": ["C-ARG1", "C-ARG0", "C-AGR2"]
}

# Change SRL labels to match dataset annotation vocabulary.
for factor in gen_factors:
    gen_factors[factor] = ["I-" + lbl if (lbl != "O") else lbl for lbl in gen_factors[factor]]


metrics = [Metric.Z_DIFF, Metric.Z_MIN_VAR, Metric.MIG, Metric.INFORMATIVENESS, Metric.COMPLETENESS]
disentang_report = DisentanglementProbe(model, dataset, sample_size=1000, metrics=metrics, gen_factors=gen_factors,
                                        annotations=annotations).report()


# Probing latent interpolation
nltk.download('punkt_tab')

sentences = [
    ("humans require freshwater for survival", "B-ARG0 B-V B-ARG1 B-ARGM-PRP I-ARGM-PRP"),
    ("animals require food to survive", "B-ARG0 B-V B-ARG1 B-ARGM-PRP I-ARGM-PRP"),
    ("the sun is in the northern hemisphere", "B-ARG0 I-ARG0 B-V B-ARGM-LOC I-ARGM-LOC I-ARGM-LOC I-ARGM-LOC"),
    ("food is a source of energy for animals / plants", "B-ARG0 B-V B-ARG2 I-ARG2 I-ARG2 I-ARG2 B-ARGM-PRP I-ARGM-PRP")
]
sentences_ds = ListImporter(annotations=["srl_f"])([[(tok, lbl) for tok, lbl in zip(sent[0].split(), sent[1].split())] for sent in sentences]).sentences

interp_dataset = [(sentences_ds[0], sentences_ds[1]), (sentences_ds[2], sentences_ds[3])]

interp_report = InterpolationProbe(model, interp_dataset, eval=[InterpMetric.SMOOTHNESS], annotations=annotations).report()
```

## How to / Tutorial

A step-by-step interactive breakdown of the quick start example and the other LangSpace probes can be found on this [Colab notebook](https://colab.research.google.com/drive/1l4JGTVYGFAPiftrnmz0SdlcUwiamR1sa). You can try them  in Colab using one of our [pre-trained models](https://huggingface.co/neuro-symbolic-ai).


## Documentation

Usage and API documentation can be found at https://langspace.readthedocs.io.


## License

LangSpace is licensed under the GPLv3 License. See the LICENSE file for details.


## Citation

If you find this work useful or use it in your research, please consider citing us

```bibtex
@inproceedings{carvalho2025langvae,
 author = {Carvalho, Danilo Silva and Zhang, Yingji and Unsworth, Harriet and Freitas, Andre},
 booktitle = {ArXiv},
 editor = {},
 pages = {0--0},
 publisher = {ArXiv},
 title = {LangVAE and LangSpace: Building and Probing for Language Model VAEs},
 volume = {0},
 year = {2025}
}
```