Quick start
===========

.. _installation:

Installation
------------

To use LangSpace, first install it using pip:

.. code-block:: console

   (.venv) $ pip install langspace


Probing an LM-VAE
------------------

.. _usage:

Here's a basic example of how to perform a disentanglement evaluation and an interpolation probe on an LM-VAE model trained with LangVAE
(or use our example `Colab notebook <https://colab.research.google.com/drive/1l4JGTVYGFAPiftrnmz0SdlcUwiamR1sa>`_):


.. code-block:: python

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

