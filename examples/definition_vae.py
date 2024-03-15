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

DEVICE = "cpu"


# Loading and evaluating pretrained model
model = LangVAE.load_from_hf_hub(models.OPTIMUS_ENTAILMENTBANK, allow_pickle=True) # Loads Optimus definition model (LangVAE) from HF.
model.eval()
model.to(DEVICE)
# wiktdefs = WiktionaryDefinitionCorpus.from_resource("pos+lemma+ctag+dep+dsr#sample") # Loads annotated dataset
eb_dataset = [sent for sent in EntailmentBankDataSet()
              if (sent.annotations["type"] == "answer" or sent.annotations["type"].startswith("context"))]
# Returns a pandas.DataFrame: cols = metrics, single row
# disentang_report = DisentanglementProbe(
#     model, wiktdefs, sample_size=1000, metrics=[Metric.Z_DIFF, Metric.MIG, ...],
#     gen_factors={"Quality": ["DIFFERENTIA-QUALITY", "QUALITY-MODIFIER", "ACCESSORY-QUALITY"], ...}).report()
# print(disentang_report)

# Returns a pandas.DataFrame: cols = dims, rows = distance, vals = generated sentences
trav_report = TraversalProbe(model, eb_dataset[:2], sample_size=10, dims=list(range(32))).report()
print(trav_report)
trav_report.to_csv("traversal.csv")

# Returns a pandas.DataFrame: cols = seeds, rows = distance from start, vals = generated sentences
interp_dataset = [
    ("humans require freshwater for survival", "animals require food to survive"),
    ("the sun is in the northern hemisphere", "food is a source of energy for animals / plants")
]
interp_report = InterpolationProbe(model, interp_dataset, eval=[InterpMetric.QUALITY, InterpMetric.SMOOTHNESS]).report()
print(interp_report)
interp_report.to_csv("interp.csv")

op_dataset = [
    ("animals require food for survival", "animals require warmth for survival"),
    ("water vapor is invisible", "the water is warm")
]
arith_report = ArithmeticProbe(model, op_dataset, ops=list(ArithmeticOps)).report()
print(arith_report)
arith_report.to_csv("arithm.csv")

cluster_viz_report = ClusterVisualizationProbe(model, [(sent.surface, sent.annotations["type"]) for sent in eb_dataset],
                                               sample_size=1000, methods=[CvM.UMAP, CvM.TSNE, CvM.PCA]).report()

