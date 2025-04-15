import math
import random
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Iterable, Dict
from copy import deepcopy
from torch import Tensor, nn
from torch.utils.data import DataLoader
from pandas import DataFrame
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from saf import Sentence
from langvae import LangVAE
from langspace.metrics.disentanglement import DisentanglementMetric
from .. import LatentSpaceProbe


class GenerativeDataset:
    """
        A base dataset class for capturing the generative factors and corresponding representations
        from a collection of sentences or samples.

        Attributes:
            generative_factors (List[Any]): A list to hold the names of generative factors.
            value_space (List[List[Any]]): For each generative factor, its associated value range or
                the unique set of factor values observed.
            sample_space (List[List[List[int]]]): For each generative factor and each value in its value_space,
                this holds the list of sentence indices (or sample indices) corresponding to that value.
            representation_space (List[Any]): A list to store extracted latent representations of sentences,
                organized based on the sample_space.
        """
    def __init__(self):
        # generative factors
        self.generative_factors = []

        # respective value range of each generative factors
        self.value_space = []

        # sentence indexes of sentences having each value
        self.sample_space = []

        # representations of sentences based on sample space
        self.representation_space = []

    def get_representation_space(self, representations):
        """
        Populate the representation_space based on the sample_space and provided latent representations.

        For each generative factor group in sample_space, the method iterates over every
        unique value and extracts the corresponding representation (row) from the given
        representations (e.g., a 2D tensor or array). The result is stored in the representation_space,
        preserving the structure of the sample_space.

        Args:
            representations (Tensor or np.ndarray): A 2D container of latent representations where each row
            corresponds to a sentence or sample.
        """
        for i in range(0, len(self.sample_space)):
            self.representation_space.append([[] for _ in range(0, len(self.sample_space[i]))])
            for j in range(0, len(self.sample_space[i])):
                self.representation_space[i][j] = representations[self.sample_space[i][j], :]


class SRLFactorDataset(GenerativeDataset):
    """
    A GenerativeDataset for organizing sentences based on Semantic Role Labeling (SRL) generative factors.

    This dataset processes a collection of sentence data along with corresponding semantic role
    annotations to extract and organize generative factors. It groups sentences by unique role
    patterns for each generative factor and records both the unique patterns (value_space) and the
    corresponding sentence indices (sample_space).

    Args:
        data (Iterable): A collection of sentence data where each element is a tuple.
            The first element is the sentence, and the second element is a list of semantic role labels.
            Example:
                [
                    ("The cat chased the mouse.", ["arg0", "v", "arg1"]),
                    ("Dogs bark loudly.", ["arg0", "v"]),
                    ...
                ]
        gen_factors (Dict[str, List[Any]]): A dictionary mapping generative factor names to lists of
            expected role values. For example:
                {"agent": ["arg0"], "patient": ["arg1"]}

    Attributes:
        generative_factors (List[str]): List of generative factor keys extracted from gen_factors.
        value_space (List[List[Any]]): For each generative factor, contains the unique role patterns
            encountered in the data.
        sample_space (List[List[List[int]]]): For each generative factor and each unique role pattern, stores
            the indices of sentences that match that pattern.
        structure (List[Any]): A list capturing, for each sentence, the generative factor structure derived
            from its semantic role labels.
    """
    def __init__(self, data, gen_factors):
        """
        Initialize the SRLFactorDataset by processing the provided sentence data and generative factor definitions.

        The constructor performs the following tasks:
          1. Initializes base attributes from GenerativeDataset.
          2. Extracts generative factor keys from the provided gen_factors and initializes the value_space
             and sample_space with lists corresponding to each factor.
          3. Constructs a dictionary mapping each role value to its corresponding generative factor.
          4. Iterates over each sentence in the data, filtering the semantic role labels that match any of the
             defined factors.
          5. For each generative factor present in a sentence, collates the corresponding role labels into a temporary list.
          6. If this role pattern has not been recorded for that factor, it is added to value_space and the current
             sentence index is recorded in sample_space. If it exists, the index is appended to the existing list.

        Args:
            data (List[List[str, List[str]]]): A collection of sentence examples where each example is a tuple.
            The first element is the sentence, and the second element is a list of semantic role labels.
            gen_factors (Dict[str, List[Any]]): A mapping of generative factor names to lists of possible role values.
        """
        super().__init__()
        dic = dict()
        self.generative_factors.extend(gen_factors.keys())
        self.value_space.extend([gen_factors[factor] for factor in self.generative_factors])
        self.sample_space.extend([[list() for value in gen_factors[factor]] for factor in self.generative_factors])

        for factor in self.generative_factors:
            for value in gen_factors[factor]:
                dic[value] = factor

        self.structure = list()
        index = 0
        for d in data:
            srl_tags = [k for k in d[1] if k in dic]
            structure = [dic[srl] for srl in srl_tags]
            for factor in self.generative_factors:
                if factor in structure:
                    temp_role = []
                    for i in range(0, len(srl_tags)):
                        if dic[srl_tags[i]] == factor:
                            temp_role.append(srl_tags[i])
                    role_index = self.generative_factors.index(factor)

                    if temp_role not in self.value_space[role_index]:
                        self.value_space[role_index].append(temp_role)
                        self.sample_space[role_index].append([index])
                    else:
                        value_index = self.value_space[role_index].index(temp_role)
                        self.sample_space[role_index][value_index].append(index)

            index += 1


class DisentanglementProbe(LatentSpaceProbe):
    """
    A probe for disentanglement metrics on the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable[Sentence], sample_size: int,
                 metrics: List[DisentanglementMetric], gen_factors: dict,
                 annotations: Dict[str, List[str]] = None, batch_size: int = 100):
        """
        Initialize the DisentanglementProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (Iterable[Sentence]): sentences to be used for the probe.

            sample_size (int): The number of data points to use for probing.
            metrics (List[DisentanglementMetric]): A list of disentanglement metrics to compute.
            gen_factors (dict): The generative factors to probe with.
            annotations(Dict[str, List[str]]): Annotation types and their respective possible values.
        """
        super(DisentanglementProbe, self).__init__(model, data, sample_size)
        self.metrics = metrics
        self.gen_factors = deepcopy(gen_factors)
        self.sample_size = sample_size
        self.annotations = annotations

        # get annotation
        first_annotation = list(annotations.keys())[0]
        ds = [[sent.surface, [tok.annotations[first_annotation] for tok in sent.tokens]] for sent in data]

        self.dataset = SRLFactorDataset(ds[:sample_size], self.gen_factors)

        # get latent representation
        sents = data[:sample_size]
        latent = self.batched_encoding(sents, annotations=annotations, batch_size=batch_size)
        representations = latent.cpu()
        self.representations = representations

        self.dataset.get_representation_space(representations)
        self.metric_method = {
            DisentanglementMetric.Z_DIFF: self.beta_vae_metric,
            DisentanglementMetric.Z_MIN_VAR: self.factor_vae_metric,
            DisentanglementMetric.MIG: self.mutual_information_gap,
            DisentanglementMetric.DISENTANGLEMENT: self.disentanglement_completeness_informativeness,
            DisentanglementMetric.COMPLETENESS: self.disentanglement_completeness_informativeness,
            DisentanglementMetric.INFORMATIVENESS: self.disentanglement_completeness_informativeness
        }

    def group_sampling(self, generative_factor, value, batch_size) -> Tensor:
        i = self.dataset.generative_factors.index(generative_factor)
        j = self.dataset.value_space[i].index(value)
        # print("index for generative factors: ", i)
        # print("index for value space", j)
        temp_space = self.dataset.representation_space[i][j]
        # print("how many sentences exist in this index: ", len(temp_space))
        # print(random.sample(range(0, temp_space.shape[0]), min(batch_size, temp_space.shape[0])))
        # print("find the latent vector (size 256): ", temp_space[random.sample(range(0, temp_space.shape[0]), min(batch_size, temp_space.shape[0])), :].shape)

        return temp_space[random.sample(range(0, temp_space.shape[0]), min(batch_size, temp_space.shape[0])), :]

    def stratified_sampling(self, generative_factor, sample_number):
        i = self.dataset.generative_factors.index(generative_factor)
        p_value = [len(self.dataset.sample_space[i][j]) for j in range(0, len(self.dataset.sample_space[i]))]

        # [0, 81, 18, 1] there are 81 sentences contain only one supertype

        samples = []
        temp = sum(p_value)
        for j in range(0, len(p_value)):
            p_value[j] = p_value[j] / temp if temp else 0
            temp_space = self.dataset.representation_space[i][j]
            """
            81 by 256
            """
            temp_sample_number = round(sample_number * p_value[j])
            temp_samples = temp_space[random.sample(range(0, temp_space.shape[0]), min(temp_sample_number, temp_space.shape[0])), :]
            """
            random_index by 256
            """
            samples.append(temp_samples)

        return samples, torch.tensor(p_value)

    @staticmethod
    def categorical_crossentropy_loss(y_pred, y_true):
        return nn.NLLLoss()(torch.log(y_pred), y_true)

    @staticmethod
    def entropy(p: Tensor):
        temp = p.flatten()
        temp = temp[temp > 0]

        return torch.sum(- temp * torch.log(temp))

    def mutual_information_estimation(self, num_bins, sample_number, normalize=False):
        z_max = self.representations.max(dim=0).values
        z_min = self.representations.min(dim=0).values

        # get the max and min of each dimension

        h_z = []
        for k in range(0, self.representations.shape[1]):
            p_z = []
            temp_z = self.representations[:, k]
            bins = z_min[k] + torch.arange(0, num_bins + 1) * (z_max[k] - z_min[k]) / num_bins
            for b in range(0, num_bins):
                if b == num_bins - 1:
                    temp = torch.where((temp_z >= bins[b]) & (temp_z <= bins[b + 1]))
                else:
                    temp = torch.where((temp_z >= bins[b]) & (temp_z < bins[b + 1]))

                # temp: the index that in this range.
                # temp[0].shape[0]: how many sentences are in this range.
                p_z.append(temp[0].shape[0])
            p_z = torch.tensor(p_z)
            p_z = p_z / torch.sum(p_z)
            h_z.append(self.entropy(p_z))

            # how to calculate the entropy from dataset??
            # - use bins. set some interval and calculate the probability z left in this range. think interval as class.
            # h_z size 256 each item is the entropy values.

        mutual_information = []
        for i in range(0, len(self.dataset.generative_factors)):
            samples, p_value = self.stratified_sampling(self.dataset.generative_factors[i], sample_number)

            # samples: [[random_index by 256], ..., [1 by 256]]
            # p_value: [0/100, 81/100, 18/100, 1/100] there are 81 sentences contain only one supertype
            h_value = None
            if normalize:
                h_value = self.entropy(p_value)

            mi = [0 for _ in range(0, self.representations.shape[1])]
            for k in range(0, self.representations.shape[1]):
                h_z_given_value = 0
                bins = z_min[k] + torch.arange(0, num_bins + 1) * (z_max[k] - z_min[k]) / num_bins
                for j in range(0, p_value.shape[0]):
                    p_z_given_value = []
                    temp_z = samples[j][:, k]
                    for b in range(0, num_bins):
                        if b == num_bins - 1:
                            temp = torch.where((temp_z >= bins[b]) & (temp_z <= bins[b + 1]))
                        else:
                            temp = torch.where((temp_z >= bins[b]) & (temp_z < bins[b + 1]))
                        # how many sentences in this bin
                        p_z_given_value.append(temp[0].shape[0])

                    p_z_given_value = torch.tensor(p_z_given_value)
                    p_z_given_value = p_z_given_value / max(1, int(p_z_given_value.sum().item()))

                    h_z_given_value += p_value[j] * self.entropy(p_z_given_value)

                # MI = H(z) - H(z|x)
                mi[k] = h_z[k] - h_z_given_value
                if normalize:
                    mi[k] = (mi[k] / h_value if h_value > 0 else 0 * mi[k])

            mutual_information.append(mi)
        return torch.tensor(mutual_information)

    def beta_vae_metric(self, batch_size=64, sample_number=50):
        initial = True
        x, y = None, None

        # sample for each label
        for i in range(0, len(self.dataset.generative_factors)):
            # sample observations for classification
            index = []
            for j in range(0, len(self.dataset.sample_space[i])):
                index = index + self.dataset.sample_space[i][j]

            if (index):
                for b in range(0, sample_number):
                    index_sample = random.sample(index, 1)[0]
                    for j in range(0, len(self.dataset.sample_space[i])):
                        if index_sample in self.dataset.sample_space[i][j]:
                            break
                    z1 = self.group_sampling(self.dataset.generative_factors[i], self.dataset.value_space[i][j], batch_size)
                    z2 = self.group_sampling(self.dataset.generative_factors[i], self.dataset.value_space[i][j], batch_size)
                    z_diff = torch.mean(torch.abs(z1 - z2), dim=0)
                    z_diff.unsqueeze_(dim=0)
                    if initial:
                        x = z_diff
                        y = i * torch.ones((1,), dtype=torch.int64)
                        initial = False
                    else:
                        x = torch.cat([x, z_diff], dim=0)
                        y = torch.cat([y, i * torch.ones((1,), dtype=torch.int64)], dim=0)

        y = F.one_hot(y)

        # randomly shuffle data
        indices = torch.randperm(x.shape[0])
        x = x[indices, :]
        y = y[indices, :]

        # split
        x_train, x_test = x[:int(0.8 * x.shape[0]), :], x[int(0.8 * x.shape[0]):, :]
        y_train, y_test = y[:int(0.8 * y.shape[0]), :], y[int(0.8 * y.shape[0]):, :]
        # print("[Beta-VAE]: training points: {:d}, test points: {:d}".format(x_train.shape[0], x_test.shape[0]))
        x_train_loader, x_test_loader = DataLoader(x_train, batch_size=64), DataLoader(x_test, batch_size=64)
        y_train_loader, y_test_loader = DataLoader(y_train, batch_size=64), DataLoader(y_test, batch_size=64)

        # 10 simple linear classifiers
        acc = torch.zeros(10)
        for i in tqdm(range(0, 10), desc="Training z-diff classifiers"):
            model = nn.Sequential(
                nn.Linear(x.shape[1], y.shape[1]),
                nn.Softmax(dim=-1)
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            accuracy = torch.tensor(0.0)
            for epoch in range(10):
                model.train()
                for batch_x_train, batch_y_train in zip(x_train_loader, y_train_loader):
                    optimizer.zero_grad()
                    y_pred = model(batch_x_train)
                    loss = self.categorical_crossentropy_loss(y_pred, batch_y_train.argmax(dim=-1))
                    loss.backward()
                    optimizer.step()

            model.eval()
            for batch_x_test, batch_y_test in zip(x_test_loader, y_test_loader):
                y_pred = model(batch_x_test)
                accuracy += (y_pred.argmax(dim=-1) == batch_y_test.argmax(dim=-1)).int().sum()

            acc[i] = accuracy / y_test.shape[0]
        # print("Beta-VAE metric score: mean: {:.2f}%, std: {:.2f}%".format(np.mean(acc) * 100, np.std(acc) * 100))
        return acc.mean(), acc.std()

    def factor_vae_metric(self, batch_size=64, sample_number=1000):

        scale = self.representations.std(dim=0)

        initial = True
        x, y = None, None

        # sample for each pos
        for i in range(0, len(self.dataset.generative_factors)):
            index = []

            for j in range(0, len(self.dataset.sample_space[i])):
                index = index + self.dataset.sample_space[i][j]

            # [[], [0, 2, 5, 6, 7, 9], [1, 4, 8], [3]] number means index of sentence
            # [0, 2, 5, 6, 7, 9, 1, 4, 8, 3]

            # print("self.dataset.sample_space: ", self.dataset.sample_space[i])
            # print("index: ", index)
            if (index):
                for b in range(0, sample_number):

                    index_sample = random.sample(index, 1)[0] # For each factor, randomly choose a sentence.
                    # print("randomly choose a sentence index: ", index_sample)

                    for j in range(0, len(self.dataset.sample_space[i])):
                        if index_sample in self.dataset.sample_space[i][j]:
                            break

                    z = self.group_sampling(self.dataset.generative_factors[i], self.dataset.value_space[i][j], batch_size)
                    z_var = (z / scale).var(dim=0)

                    if initial:
                        x = z_var.argmin() * torch.ones((1,))
                        # print("index (256) corresponding to smallest var: ", x)
                        # exit()
                        y = i * torch.ones((1,), dtype=torch.int64)
                        initial = False
                    else:
                        x = torch.cat([x, z_var.argmin() * torch.ones((1,), dtype=torch.int64)], dim=0)
                        y = torch.cat([y, i * torch.ones((1,), dtype=torch.int64)], dim=0)


        # 10 majority vote classifiers
        acc = []
        for i in range(0, 10):
            indices = torch.randperm(x.shape[0])
            x = x[indices]
            y = y[indices]
            x_train, x_test = x[:int(0.8 * x.shape[0])], x[int(0.8 * x.shape[0]):]
            y_train, y_test = y[:int(0.8 * y.shape[0])], y[int(0.8 * y.shape[0]):]
            V = torch.zeros((self.representations.shape[1], len(self.dataset.generative_factors)))
            for j in range(0, x_train.shape[0]):
                V[int(x_train[j]), int(y_train[j])] += 1
            temp = 0
            for j in range(0, x_test.shape[0]):
                if V[int(x_test[j]), :].argmax() == y_test[j]:
                    temp += 1
            acc.append(temp / x_test.shape[0])
        acc = torch.tensor(acc)
        # print("Factor-VAE metric score: mean: {:.2f}%, std: {:.2f}%".format(np.mean(acc) * 100, np.std(acc) * 100))
        return acc.mean(), acc.std()

    def mutual_information_gap(self, num_bins=20, sample_number=10000):
        mi = self.mutual_information_estimation(num_bins, sample_number, normalize=True)
        mig = []
        for i in range(0, mi.shape[0]): # 7
            temp_mi = mi[i, :].tolist()
            temp_mi.sort(reverse=True)
            mig.append(temp_mi[0] - temp_mi[1])
        # print("Mutual Information Gap: {:.4f}".format(sum(mig) / len(mig)))
        return torch.tensor(mig).mean(), torch.tensor(mig).std()

    def modularity_explicitness(self, num_bins=20, sample_number=10000):
        mi = self.mutual_information_estimation(num_bins, sample_number) # 7 by 256
        mask = torch.zeros(mi.shape)
        index = mi.argmax(dim=0) # 256

        for i in range(0, index.shape[0]):
            mask[index[i], i] = 1
        temp_t = mi * mask

        # first remove the factor with the biggest MI for each dimension.
        # calculate variance of each dimension (mu is 0) of remaining factors.

        delta = (mi - temp_t).square().sum(dim=0) / (temp_t.square().sum(dim=0) * (mi.shape[0] - 1))
        modularity = 1 - delta
        x_train, x_test, y_train, y_test = None, None, None, None
        # print("Modularity: {:.4f}".format(np.mean(modularity)))

        explicitness = []
        for i in range(0, len(self.dataset.generative_factors)):
            samples = self.stratified_sampling(self.dataset.generative_factors[i], sample_number)[0]
            for j in range(0, len(samples)):
                temp = samples[j]
                temp_train, temp_test = temp[:int(0.8 * temp.shape[0]), :], temp[int(0.8 * temp.shape[0]):, :]
                if j == 0:
                    x_train, x_test = temp_train, temp_test
                    y_train, y_test = (j * torch.ones(temp_train.shape[0], dtype=torch.int64),
                                       j * torch.ones(temp_test.shape[0], dtype=torch.int64))
                else:
                    x_train = torch.cat([x_train, temp_train], dim=0)
                    x_test = torch.cat([x_test, temp_test], dim=0)
                    y_train = torch.cat([y_train, j * torch.ones(temp_train.shape[0])], dim=0)
                    y_test = torch.cat([y_test, j * torch.ones(temp_test.shape[0])], dim=0)

            indices = torch.randperm(x_train.shape[0])
            x_train = x_train[indices, :]
            y_train = y_train[indices]

            # suggested in code from original paper
            if (y_train.sum() != 0):
                model = LogisticRegression(C=1e10, solver='liblinear')
                model.fit(x_train.numpy(), y_train.numpy())
                preds = model.predict_proba(x_test)
                roc_auc = []
                for j in range(0, len(model.classes_)):
                    y_true = (y_test == j)
                    y_pred = preds[:, j]
                    if (True in y_true):
                        roc_auc.append(roc_auc_score(y_true, y_pred))
                roc_auc = torch.tensor(roc_auc)
                explicitness.append(roc_auc.mean())

        explicitness = torch.tensor(explicitness)
        # print("Explicitness: {:.4f}".format(np.mean(explicitness)))
        return modularity.mean(), explicitness.mean()

    def disentanglement_completeness_informativeness(self, sample_number=10000):
        informativeness = []
        r = []
        disentanglement = []
        completeness = []
        for i in range(0, len(self.dataset.generative_factors)):
            samples = self.stratified_sampling(self.dataset.generative_factors[i], sample_number)[0]
            x_train, x_test, y_train, y_test = None, None, None, None

            for j in range(0, len(samples)):
                temp = samples[j]
                temp_train, temp_test = (temp[:int(np.ceil(0.8 * temp.shape[0])), :],
                                         temp[int(np.ceil(0.8 * temp.shape[0])):, :])

                # num by 256
                if j == 0:
                    x_train, x_test = temp_train, temp_test
                    y_train, y_test = (j * torch.ones(temp_train.shape[0], dtype=torch.int64),
                                       j * torch.ones(temp_test.shape[0], dtype=torch.int64))
                else:
                    x_train = torch.cat([x_train, temp_train], dim=0)
                    x_test = torch.cat([x_test, temp_test], dim=0)
                    y_train = torch.cat([y_train, j * torch.ones(temp_train.shape[0], dtype=torch.int64)], dim=0)
                    y_test = torch.cat([y_test, j * torch.ones(temp_test.shape[0], dtype=torch.int64)], dim=0)

            indices = torch.randperm(x_train.shape[0])
            x_train = x_train[indices, :]
            y_train = y_train[indices]

            # print(x_train.shape)

            # for each factor:
            if x_train.shape[0] == 0 or x_test.shape[0] == 0:
                continue

            model = RandomForestClassifier(n_estimators=10)

            model.fit(x_train.numpy(), y_train.numpy())
            informativeness.append(model.score(x_test.numpy(), y_test.numpy()))
            r.append(model.feature_importances_)

            # print(model.feature_importances_.shape) 256

        r = torch.tensor(np.stack(r))

        for i in range(0, r.shape[1]):
            p = r[:, i]
            p = p / p.sum() if (p.sum() > 1e-7) else torch.zeros(p.shape)
            h_k_p = self.entropy(p) / math.log(r.shape[0])
            disentanglement.append(1 - h_k_p)

        disentanglement = torch.tensor(disentanglement)
        weight = r.sum(dim=0) / r.sum() if (r.sum() > 1e-7) else torch.zeros(r.shape)
        # print("Disentanglement Score: {:.4f}".format(np.sum(weight * disentanglement)))

        for j in range(0, r.shape[0]):
            p = r[j, :]
            p = p / p.sum() if (p.sum() > 1e-7) else torch.zeros(p.shape)
            h_d_p = self.entropy(p) / math.log(r.shape[1])
            completeness.append(1 - h_d_p)

        completeness = torch.tensor(completeness)
        # print("Completeness Score: {:.4f}".format(np.mean(completeness)))

        informativeness = torch.tensor(informativeness)
        # print("Informativeness Score: {:.4f}".format(np.mean(informativeness)))

        return {
            # DisentanglementMetric.DISENTANGLEMENT: (np.sum(weight * disentanglement), 0),
            DisentanglementMetric.COMPLETENESS: (completeness.mean(), completeness.std()),
            DisentanglementMetric.INFORMATIVENESS: (informativeness.mean(), informativeness.std())
        }

    def separated_attribute_predictability(self, sample_number=10000):
        sap = []
        for i in range(0, len(self.dataset.generative_factors)):
            samples = self.stratified_sampling(self.dataset.generative_factors[i], sample_number)[0]
            x_train, x_test, y_train, y_test = None, None, None, None

            for j in range(0, len(samples)):
                temp = samples[j]
                temp_train, temp_test = temp[:int(0.8 * temp.shape[0]), :], temp[int(0.8 * temp.shape[0]):, :]
                if j == 0:
                    x_train, x_test = temp_train, temp_test
                    y_train, y_test = (j * torch.ones(temp_train.shape[0], dtype=torch.int64),
                                       j * torch.ones(temp_test.shape[0], dtype=torch.int64))
                else:
                    x_train = torch.cat([x_train, temp_train], dim=0)
                    x_test = torch.cat([x_test, temp_test], dim=0)
                    y_train = torch.cat([y_train, j * torch.ones(temp_train.shape[0], dtype=torch.int64)], dim=0)
                    y_test = torch.cat([y_test, j * torch.ones(temp_test.shape[0], dtype=torch.int64)], dim=0)
            indices = torch.randperm(x_train.shape[0])
            x_train = x_train[indices, :]
            y_train = y_train[indices]

            if (y_train.sum() != 0):
                acc = []
                for j in range(0, x_train.shape[1]):
                    temp_x_train, temp_x_test = x_train[:, j].reshape(-1, 1), x_test[:, j].reshape(-1, 1)
                    model = LinearSVC(C=0.01)
                    model.fit(temp_x_train.numpy(), y_train.numpy())
                    acc.append(model.score(temp_x_test.numpy(), y_test.numpy()))
                acc.sort(reverse=True)
                sap.append(acc[0] - acc[1])
        # print("SAP score: {:.4f}".format(np.mean(sap)))
        return torch.tensor(sap).mean()

    def report(self) -> DataFrame:
        """
        Generate a report from the probe.

        Returns:
            DataFrame: The generated report.
        """
        results = dict()
        calculated = set()
        for metric in self.metrics:
            if (metric in [DisentanglementMetric.Z_DIFF, DisentanglementMetric.Z_MIN_VAR, DisentanglementMetric.MIG]):
                mean, std = self.metric_method[metric]()
                results[metric.value] = f"{mean:.2f} (±{std:.2f})"
                calculated.add(metric)
            elif (metric in [DisentanglementMetric.DISENTANGLEMENT,
                             DisentanglementMetric.COMPLETENESS,
                             DisentanglementMetric.INFORMATIVENESS]
                  and metric not in calculated):
                dci_res = self.metric_method[metric]()
                for m_res in dci_res:
                    results[m_res.value] = f"{dci_res[m_res][0]:.2f} (±{dci_res[m_res][1]:.2f})"
                    calculated.add(m_res)
            else:
                pass

        return pd.DataFrame.from_records([results])


