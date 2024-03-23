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
from langspace.probe.cluster_vis import ClusterVisualizationMethod as CvM
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
eb_dataset = [sent.surface for sent in EntailmentBankDataSet()
              if (sent.annotations["type"] == "answer" or sent.annotations["type"].startswith("context"))]

# ------------------- latent Disentanglement -----------------------
# Returns a pandas.DataFrame: cols = metrics, single row
dis_list = [[s.replace('\n', '').split('&')[0].strip(), s.replace('\n', '').split('&')[1].strip()] for s in sents]

disentang_report = DisentanglementProbe(
    model, dis_list, sample_size=1000, metrics=["z-diff", "z-min-var", "MIG", "Disentanglement", "Modularity"],
    gen_factors=gen_factors).report()

# ------------------- latent visualization -------------------------
viz_list = [(s.replace('\n', '').split('&')[0].strip(), s.replace('\n', '').split('&')[1].strip()) for s in sents]

sample_size, TopK = 1000, 5

def structure_viz(viz_list, sample_size=1000, TopK=5):
    """
    semantic role structure visualization
    only show the structure (remove repeated semantic role for each sentence). E.g., ARG0 ARG0 ARG0 V ARG1 ARG1 -> ARG0 V ARG1
    arguments:
    viz_list = [[sent, semantic role labels], [], ..., []]
    sample_size
    TopK
    """
    final_viz_list = []
    for pair in viz_list:
        sent, label = pair[0], pair[1]
        unique_label = []
        for tkn in label.split(' '):
            if len(unique_label) == 0:
                unique_label.append(tkn)
            else:
                if unique_label[-1] == tkn:
                    continue
                else:
                    unique_label.append(tkn)
        final_viz_list.append((sent, ' '.join(unique_label)))

    second_values = [sublist[1] for sublist in final_viz_list]

    # Count the occurrences of label in dataset and only choose TopK as target labels.
    count_dict = Counter(second_values)
    sorted_counts = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    target_label_dict = dict([(i[0], 0) for i in sorted_counts[:TopK]])

    # make the input list balance.
    target_viz_list = []
    for i in final_viz_list:
        if i[1] in target_label_dict and target_label_dict[i[1]] <= sample_size/TopK:
            target_viz_list.append(i)
            target_label_dict[i[1]] += 1

    return target_viz_list

def role_content_viz(viz_list, target_role, sample_size=1000, TopK=5):
    # 1. count unique role-content
    role_content_dict = dict()
    target_viz_list = []
    for pair in viz_list:
        sents, labels = pair[0].split(' '), pair[1].split(' ')
        for idx, tkn in enumerate(sents):
            label = labels[idx]
            key = label + ' : ' + tkn
            if key not in target_role:
                continue
            else:
                target_viz_list.append((pair[0], key))

    return target_viz_list

# target_role = ['ARG0 : animal', 'ARG0 : water', 'ARG0 : plant', 'ARG0 : something']
target_viz_list = role_content_viz(viz_list, ['ARG0 : animal', 'ARG0 : water', 'ARG0 : plant', 'ARG0 : something'], sample_size=1000, TopK=5)
cluster_viz_report = ClusterVisualizationProbe(model, target_viz_list, sample_size=sample_size, method=[CvM.TSNE]).report()

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



