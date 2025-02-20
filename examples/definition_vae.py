# 1. Load annotated data from saf_datasets.
# 2. Load definition VAE for generation.
# 3. Run qualitative evaluation: clustering
# 3. Run qualitative evaluation: interp
import langspace.models as models
from saf.importers import ListImporter
from saf_datasets import WiktionaryDefinitionCorpus, EntailmentBankDataSet
from langvae import LangVAE
from langspace.probe import DisentanglementProbe, TraversalProbe, InterpolationProbe, ClusterVisualizationProbe, ArithmeticProbe
from langspace.metrics.disentanglement import DisentanglementMetric as Metric
from langspace.metrics.interpolation import InterpolationMetric as InterpMetric
from langspace.probe.cluster_vis.methods import ClusterVisualizationMethod as CvM
from langspace.ops.arithmetic import ArithmeticOps
import random
from collections import Counter
DEVICE = "cuda"

with open('explanations.txt') as file:
    expl_srl = [line.split(" & ") for line in file.readlines()]
    expl = [es[0].split() for es in expl_srl]
    srl = [es[1].split() for es in expl_srl]
    sents = ListImporter(annotations=["srl"])([list(zip(token, label)) for token, label in zip(expl, srl)]).sentences

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

# Loading and evaluating pretrained model
model = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langcvae-bert-base-cased-gpt2-srl-l128") # Loads Optimus definition model (LangVAE) from HF.
model.eval()
# model.to(DEVICE)

# wiktdefs = WiktionaryDefinitionCorpus.from_resource("pos+lemma+ctag+dep+dsr#sample") # Loads annotated dataset
eb_dataset = EntailmentBankDataSet.from_resource("pos+lemma+ctag+dep+srl#noproof")
dataset = [
    sent for sent in eb_dataset
    if (sent.annotations["type"] == "answer" or sent.annotations["type"].startswith("context"))
]

for factor in gen_factors:
    gen_factors[factor] = ["I-" + lbl if (lbl != "O") else lbl for lbl in gen_factors[factor]]

annotations = {"srl_f": eb_dataset.annotations["srl"]}
for sent in dataset:
    for token in sent.tokens:
        srl = token.annotations["srl"]
        token_annot = [lbl for lbl in srl if (lbl != "O")][0] if (len(set(srl)) > 1) else srl[0]
        token.annotations["srl_f"] = token_annot
#
# annotations = {"srl_f": [lbl.replace("B-", "").replace("I-", "") for lbl in eb_dataset.annotations["srl"]]}



# ------------------- latent Disentanglement -----------------------
# Returns a pandas.DataFrame: cols = metrics, single row
# dis_list = [[s.replace('\n', '').split('&')[0].strip(), s.replace('\n', '').split('&')[1].strip()] for s in sents]
#
metrics = [Metric.Z_DIFF, Metric.Z_MIN_VAR, Metric.MIG, Metric.INFORMATIVENESS, Metric.COMPLETENESS]
disentang_report = DisentanglementProbe(model, dataset, sample_size=10000, metrics=metrics, gen_factors=gen_factors,
                                        annotations=annotations).report()

print(disentang_report)
disentang_report.to_csv("disentanglement.csv")


# ------------------- latent visualization -------------------------
# #viz_list = [(s.replace('\n', '').split('&')[0].strip(), s.replace('\n', '').split('&')[1].strip()) for s in sents]

target_roles = {"ARG0": ["animal", "water", "plant", "something"]}
labels = list(target_roles.keys())
label_map = dict()
for prefix in ["B-", "I-"]:
    for lbl in labels:
        target_roles[prefix + lbl] = target_roles[lbl]
        label_map[prefix + lbl] = lbl

cluster_viz_report = ClusterVisualizationProbe(model, dataset, sample_size=1000, target_roles=target_roles,
                                               methods=[CvM.TSNE, CvM.PCA], cluster_annotation="srl_f",
                                               annotations=annotations, plot_label_map=label_map).report()


# ------------------- latent traversal ---------------------------
# Returns a pandas.DataFrame: cols = dims, rows = distance, vals = generated sentences
trav_dataset = dataset[:2]
trav_report = TraversalProbe(model, trav_dataset, sample_size=10, dims=list(range(128)), annotations=annotations).report()
print(trav_report)
trav_report.to_csv("traversal.csv")

# ------------------- latent interpolation -----------------------
# Returns a pandas.DataFrame: cols = seeds, rows = distance from start, vals = generated sentences
# interp_dataset = [
#     ("humans require freshwater for survival", "animals require food to survive"),
#     ("the sun is in the northern hemisphere", "food is a source of energy for animals / plants")
# ]
# random.shuffle(eb_dataset)
interp_dataset = [(dataset[i], dataset[i+1]) for i in range(0, 50, 2)]
interp_report = InterpolationProbe(model, interp_dataset, eval=[InterpMetric.QUALITY, InterpMetric.SMOOTHNESS],
                                   annotations=annotations).report()
print(interp_report)
interp_report.to_csv("interp.csv")

# ------------------- latent arithmetic ---------------------------
# op_dataset = [
#     ("animals require food for survival", "animals require warmth for survival"),
#     ("water vapor is invisible", "the water is warm")
# ]
# random.shuffle(eb_dataset)
op_dataset = [(dataset[i], dataset[i+1]) for i in range(0, 50, 2)]
arith_report = ArithmeticProbe(model, op_dataset, ops=list(ArithmeticOps), annotations=annotations).report()
print(arith_report)
arith_report.to_csv("arithm.csv")



