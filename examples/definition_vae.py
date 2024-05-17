# 1. Load annotated data from saf_datasets.
# 2. Load definition VAE for generation.
# 3. Run qualitative evaluation: clustering
# 3. Run qualitative evaluation: interp
import langspace.models as models
from saf_datasets import WiktionaryDefinitionCorpus, EntailmentBankDataSet
from langvae import LangVAE
from langspace.probe import DisentanglementProbe, TraversalProbe, InterpolationProbe, ClusterVisualizationProbe, ArithmeticProbe
from langspace.metrics.disentanglement import DisentanglementMetric as Metric
from langspace.metrics.interpolation import InterpolationMetric as InterpMetric
from langspace.probe.cluster_vis.methods import ClusterVisualizationMethod as CvM
from langspace.ops.arithmetic import ArithmeticOps
import random
from collections import Counter
DEVICE = "cpu"

with open('explanations.txt') as file:
    sents = file.readlines()

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
model = LangVAE.load_from_hf_hub(models.OPTIMUS_ENTAILMENTBANK, allow_pickle=True) # Loads Optimus definition model (LangVAE) from HF.
model.eval()
model.to(DEVICE)

# wiktdefs = WiktionaryDefinitionCorpus.from_resource("pos+lemma+ctag+dep+dsr#sample") # Loads annotated dataset
eb_dataset = [sent for sent in EntailmentBankDataSet.from_resource("pos+lemma+ctag+dep+srl#noproof")
              if (sent.annotations["type"] == "answer" or sent.annotations["type"].startswith("context"))]

for sent in eb_dataset:
    for token in sent.tokens:
        token.annotations["srl_0"] = token.annotations["srl"][0].replace("B-", "").replace("I-", "")

from saf import Vocabulary
vocab = Vocabulary(eb_dataset, source="srl_0", lowercase=False)

# ------------------- latent Disentanglement -----------------------
# Returns a pandas.DataFrame: cols = metrics, single row
dis_list = [[s.replace('\n', '').split('&')[0].strip(), s.replace('\n', '').split('&')[1].strip()] for s in sents]

metrics = [Metric.Z_DIFF, Metric.Z_MIN_VAR, Metric.MIG, Metric.INFORMATIVENESS, Metric.COMPLETENESS]
disentang_report = DisentanglementProbe(model, eb_dataset, sample_size=20000, metrics=metrics, gen_factors=gen_factors,
                                        annotation="srl_0").report()

print(disentang_report)
disentang_report.to_csv("disentanglement.csv")

exit(0)


# ------------------- latent visualization -------------------------
# viz_list = [(s.replace('\n', '').split('&')[0].strip(), s.replace('\n', '').split('&')[1].strip()) for s in sents]
# sample_size, TopK = 1000, 5
#
# # target_role = ['ARG0 : animal', 'ARG0 : water', 'ARG0 : plant', 'ARG0 : something']
# target_viz_list = ClusterVisualizationProbe.role_content_viz(viz_list, ['ARG0 : animal', 'ARG0 : water', 'ARG0 : plant', 'ARG0 : something'], sample_size=1000, TopK=5)
# cluster_viz_report = ClusterVisualizationProbe(model, target_viz_list, sample_size=sample_size, methods=[CvM.TSNE]).report()


# ------------------- latent traversal ---------------------------
# Returns a pandas.DataFrame: cols = dims, rows = distance, vals = generated sentences
trav_dataset = eb_dataset[:2]
trav_report = TraversalProbe(model, trav_dataset, sample_size=10, dims=list(range(32))).report()
print(trav_report)
trav_report.to_csv("traversal.csv")

# ------------------- latent interpolation -----------------------
# Returns a pandas.DataFrame: cols = seeds, rows = distance from start, vals = generated sentences
# interp_dataset = [
#     ("humans require freshwater for survival", "animals require food to survive"),
#     ("the sun is in the northern hemisphere", "food is a source of energy for animals / plants")
# ]
# random.shuffle(eb_dataset)
interp_dataset = [(eb_dataset[i], eb_dataset[i+1]) for i in range(0, 50, 2)]
interp_report = InterpolationProbe(model, interp_dataset, eval=[InterpMetric.QUALITY, InterpMetric.SMOOTHNESS]).report()
print(interp_report)
interp_report.to_csv("interp.csv")

# ------------------- latent arithmetic ---------------------------
# op_dataset = [
#     ("animals require food for survival", "animals require warmth for survival"),
#     ("water vapor is invisible", "the water is warm")
# ]
# random.shuffle(eb_dataset)
op_dataset = [(eb_dataset[i], eb_dataset[i+1]) for i in range(0, 50, 2)]
arith_report = ArithmeticProbe(model, op_dataset, ops=list(ArithmeticOps)).report()
print(arith_report)
arith_report.to_csv("arithm.csv")



