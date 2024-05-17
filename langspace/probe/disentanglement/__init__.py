from typing import List, Iterable, Union

import pandas as pd
from pandas import DataFrame
from saf import Sentence
from langvae import LangVAE
from langspace.metrics.disentanglement import DisentanglementMetric
from .. import LatentSpaceProbe
import torch.nn.functional as F
import numpy as np
import random
import tensorflow as tf
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm

class GenerativeDataset:
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
        for i in range(0, len(self.sample_space)):
            self.representation_space.append([[] for _ in range(0, len(self.sample_space[i]))])
            for j in range(0, len(self.sample_space[i])):
                self.representation_space[i][j] = representations[self.sample_space[i][j], :]


class SRLFactorDataset(GenerativeDataset):
    def __init__(self, data, gen_factors):
        """
        role_list: list of the annotation of sentences, each annotation is a string, the role of each word is separated by a space.
        E.g., ["arg0 v arg1", ... , "arg0 arg0 v arg1 arg1 arg1"]
        """
        super().__init__()
        dic = {}
        self.generative_factors.extend(gen_factors.keys())
        self.value_space.extend([gen_factors[factor] for factor in self.generative_factors])
        self.sample_space.extend([[list() for value in gen_factors[factor]] for factor in self.generative_factors])

        for factor in self.generative_factors:
            for value in gen_factors[factor]:
                dic[value] = factor

        self.structure = list()
        index = 0
        for d in data:
            srl_tags = [k for k in d[1].strip().split(' ') if k in dic]
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


class DisentanglementProbe(LatentSpaceProbe):
    """
    Class for probing disentanglement metrics for the latent space of a language VAE.
    """
    def __init__(self, model: LangVAE, data: Iterable[Union[List[str], Sentence]], sample_size: int,
                 metrics: List[DisentanglementMetric], gen_factors: dict, annotation: str, batch_size: int = 100):
        """
        Initialize the DisentanglementProbe.

        Args:
            model (LangVAE): The language model to probe.
            data (Iterable[Union[str, Sentence]]):
                [['animals require food for survival', 'arg0 v arg1 prp prp'], ..., ]

            sample_size (int): The number of data points to use for probing.
            metrics (List[DisentanglementMetric]): A list of disentanglement metrics to compute.
            gen_factors (dict): The generative factors to probe with.
        """
        super(DisentanglementProbe, self).__init__(model, data, sample_size)
        self.metrics = metrics
        self.gen_factors = gen_factors
        self.sample_size = sample_size
        self.annotation = annotation

        # get annotation
        if (isinstance(list(data[:1])[0], Sentence)):
            data = [[sent.surface, " ".join([tok.annotations[annotation] for tok in sent.tokens])]
                    for sent in data]

        self.dataset = SRLFactorDataset(data[:sample_size], gen_factors)

        # get latent representation
        sents = [sent[0].strip() for sent in data[:sample_size]]
        latent = self.encoding(sents, batch_size=batch_size)
        representations = latent.numpy()
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

    # def encoding(self, data):
    #     """
    #             Encodes the sentences
    #
    #             Args:
    #                 data (List[str]): sentences
    #
    #             Returns:
    #                 Tensor: Latent representation
    #             """
    #     seed = list(data)
    #     if (len(seed) < 2):
    #         seed.append("")
    #
    #     encode_seed = self.model.decoder.tokenizer(seed, padding="max_length", truncation=True,
    #                                                max_length=self.model.decoder.max_len, return_tensors='pt')
    #     encode_seed_oh = F.one_hot(encode_seed["input_ids"],
    #                                num_classes=len(self.model.decoder.tokenizer.get_vocab())).to(torch.int8)
    #     with torch.no_grad():
    #         latent = self.model.encode_z(encode_seed_oh)
    #
    #     return latent

    def group_sampling(self, generative_factor, value, batch_size):
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

        return samples, np.array(p_value)

    @ staticmethod
    def entropy(p):
        temp = p.flatten()
        temp = temp[np.where(temp > 0)]

        return np.sum(- temp * np.log(temp))

    def mutual_information_estimation(self, num_bins, sample_number, normalize=False):
        z_max = np.max(self.representations, axis=0)
        z_min = np.min(self.representations, axis=0)

        # get the max and min of each dimension

        h_z = []
        for k in range(0, self.representations.shape[1]):
            p_z = []
            temp_z = self.representations[:, k]
            bins = z_min[k] + np.arange(0, num_bins + 1) * (z_max[k] - z_min[k]) / num_bins
            for b in range(0, num_bins):
                if b == num_bins - 1:
                    temp = np.where((temp_z >= bins[b]) & (temp_z <= bins[b + 1]))
                else:
                    temp = np.where((temp_z >= bins[b]) & (temp_z < bins[b + 1]))

                # temp: the index that in this range.
                # temp[0].shape[0]: how many sentences are in this range.
                p_z.append(temp[0].shape[0])
            p_z = np.array(p_z)
            p_z = p_z / np.sum(p_z)
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
                bins = z_min[k] + np.arange(0, num_bins + 1) * (z_max[k] - z_min[k]) / num_bins
                for j in range(0, p_value.shape[0]):
                    p_z_given_value = []
                    temp_z = samples[j][:, k]
                    for b in range(0, num_bins):
                        if b == num_bins - 1:
                            temp = np.where((temp_z >= bins[b]) & (temp_z <= bins[b + 1]))
                        else:
                            temp = np.where((temp_z >= bins[b]) & (temp_z < bins[b + 1]))
                        # how many sentences in this bin
                        p_z_given_value.append(temp[0].shape[0])

                    p_z_given_value = np.array(p_z_given_value)
                    p_z_given_value = p_z_given_value / max(1, int(np.sum(p_z_given_value)))

                    h_z_given_value += p_value[j] * self.entropy(p_z_given_value)

                # MI = H(z) - H(z|x)
                mi[k] = h_z[k] - h_z_given_value
                if normalize:
                    mi[k] = (mi[k] / h_value if h_value > 0 else 0 * mi[k])

            mutual_information.append(mi)
        return np.array(mutual_information)

    def beta_vae_metric(self, batch_size=64, sample_number=50):
        initial = True
        x, y = None, None

        # sample for each pos
        for i in range(0, len(self.dataset.generative_factors)):
            # sample observations for classification
            index = []
            for j in range(0, len(self.dataset.sample_space[i])):
                index = index + self.dataset.sample_space[i][j]

            for b in range(0, sample_number):
                index_sample = random.sample(index, 1)[0]
                for j in range(0, len(self.dataset.sample_space[i])):
                    if index_sample in self.dataset.sample_space[i][j]:
                        break
                z1 = self.group_sampling(self.dataset.generative_factors[i], self.dataset.value_space[i][j], batch_size)
                z2 = self.group_sampling(self.dataset.generative_factors[i], self.dataset.value_space[i][j], batch_size)
                z_diff = np.mean(np.abs(z1 - z2), axis=0)
                z_diff.resize((1, z_diff.shape[0]))
                if initial:
                    x = z_diff
                    y = i * np.ones(shape=(1,))
                    initial = False
                else:
                    x = np.concatenate([x, z_diff], axis=0)
                    y = np.concatenate([y, i * np.ones(shape=(1,))], axis=0)

        y = tf.keras.utils.to_categorical(y)

        # randomly shuffle data
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices, :]
        y = y[indices, :]

        # split
        x_train, x_test = x[:int(0.8 * x.shape[0]), :], x[int(0.8 * x.shape[0]):, :]
        y_train, y_test = y[:int(0.8 * y.shape[0]), :], y[int(0.8 * y.shape[0]):, :]
        print("[Beta-VAE]: training points: {:d}, test points: {:d}".format(x_train.shape[0], x_test.shape[0]))

        # 10 simple linear classifiers
        acc = []
        for i in range(0, 10):
            inputs = tf.keras.Input(shape=(x.shape[1],))
            outputs = tf.keras.layers.Dense(y.shape[1], activation='softmax')(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                          loss="categorical_crossentropy", metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2, verbose=0)
            test_scores = model.evaluate(x_test, y_test, verbose=0)
            acc.append(test_scores[1])
        acc = np.array(acc)
        # print("Beta-VAE metric score: mean: {:.2f}%, std: {:.2f}%".format(np.mean(acc) * 100, np.std(acc) * 100))
        return np.mean(acc), np.std(acc)

    def factor_vae_metric(self, batch_size=64, sample_number=1000):

        scale = np.std(self.representations, axis=0)

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
            for b in range(0, sample_number):

                index_sample = random.sample(index, 1)[0] # For each factor, randomly choose a sentence.
                # print("randomly choose a sentence index: ", index_sample)

                for j in range(0, len(self.dataset.sample_space[i])):
                    if index_sample in self.dataset.sample_space[i][j]:
                        break

                # print("find sample space index: ", j)
                # print(self.dataset.generative_factors[i])
                # print(self.dataset.value_space[i][j])
                # print(batch_size)

                z = self.group_sampling(self.dataset.generative_factors[i], self.dataset.value_space[i][j], batch_size)

                # print("scale: ", scale.shape)
                # print("z shape: ", z.shape)
                # exit()
                z_var = np.var(z / scale, axis=0)
                # print("z_var shape: ", z_var.shape)

                # print("z_var.shape: ", z_var.shape)

                if initial:
                    x = np.argmin(z_var) * np.ones(shape=(1,))
                    # print("index (256) corresponding to smallest var: ", x)
                    # exit()
                    y = i * np.ones(shape=(1,))
                    initial = False
                else:
                    x = np.concatenate([x, np.argmin(z_var) * np.ones(shape=(1,))], axis=0)
                    y = np.concatenate([y, i * np.ones(shape=(1,))], axis=0)


        # 10 majority vote classifiers
        acc = []
        for i in range(0, 10):
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]
            x_train, x_test = x[:int(0.8 * x.shape[0])], x[int(0.8 * x.shape[0]):]
            y_train, y_test = y[:int(0.8 * y.shape[0])], y[int(0.8 * y.shape[0]):]
            V = np.zeros(shape=(self.representations.shape[1], len(self.dataset.generative_factors)))
            for j in range(0, x_train.shape[0]):
                V[int(x_train[j]), int(y_train[j])] += 1
            temp = 0
            for j in range(0, x_test.shape[0]):
                if np.argmax(V[int(x_test[j]), :]) == y_test[j]:
                    temp += 1
            acc.append(temp / x_test.shape[0])
        acc = np.array(acc)
        # print("Factor-VAE metric score: mean: {:.2f}%, std: {:.2f}%".format(np.mean(acc) * 100, np.std(acc) * 100))
        return np.mean(acc), np.std(acc)

    def mutual_information_gap(self, num_bins=20, sample_number=10000):
        mi = self.mutual_information_estimation(num_bins, sample_number, normalize=True)
        mig = []
        for i in range(0, mi.shape[0]): # 7
            temp_mi = mi[i, :].tolist()
            temp_mi.sort(reverse=True)
            mig.append(temp_mi[0] - temp_mi[1])
        # print("Mutual Information Gap: {:.4f}".format(sum(mig) / len(mig)))
        return np.mean(mig), np.std(mig)

    def modularity_explicitness(self, num_bins=20, sample_number=10000):
        mi = self.mutual_information_estimation(num_bins, sample_number) # 7 by 256
        mask = np.zeros(shape=mi.shape)
        index = np.argmax(mi, axis=0) # 256

        for i in range(0, index.shape[0]):
            mask[index[i], i] = 1
        temp_t = mi * mask

        # first remove the factor with the biggest MI for each dimension.
        # calculate variance of each dimension (mu is 0) of remaining factors.

        delta = np.sum(np.square(mi - temp_t), axis=0) / (np.sum(np.square(temp_t), axis=0) * (mi.shape[0] - 1))
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
                    y_train, y_test = j * np.ones(temp_train.shape[0]), j * np.ones(temp_test.shape[0])
                else:
                    x_train = np.concatenate([x_train, temp_train], axis=0)
                    x_test = np.concatenate([x_test, temp_test], axis=0)
                    y_train = np.concatenate([y_train, j * np.ones(temp_train.shape[0])], axis=0)
                    y_test = np.concatenate([y_test, j * np.ones(temp_test.shape[0])], axis=0)
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices, :]
            y_train = y_train[indices]

            # suggested in code from original paper
            if (np.sum(y_train) != 0):
                model = LogisticRegression(C=1e10, solver='liblinear')
                model.fit(x_train, y_train)
                preds = model.predict_proba(x_test)
                roc_auc = []
                for j in range(0, len(model.classes_)):
                    y_true = (y_test == j)
                    y_pred = preds[:, j]
                    if (True in y_true):
                        roc_auc.append(roc_auc_score(y_true, y_pred))
                roc_auc = np.array(roc_auc)
                explicitness.append(np.mean(roc_auc))

        explicitness = np.array(explicitness)
        # print("Explicitness: {:.4f}".format(np.mean(explicitness)))
        return np.mean(modularity), np.mean(explicitness)

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
                temp_train, temp_test = temp[:int(np.ceil(0.8 * temp.shape[0])), :], temp[int(np.ceil(0.8 * temp.shape[0])):, :]

                # num by 256
                if j == 0:
                    x_train, x_test = temp_train, temp_test
                    y_train, y_test = j * np.ones(temp_train.shape[0]), j * np.ones(temp_test.shape[0])
                else:
                    x_train = np.concatenate([x_train, temp_train], axis=0)
                    x_test = np.concatenate([x_test, temp_test], axis=0)
                    y_train = np.concatenate([y_train, j * np.ones(temp_train.shape[0])], axis=0)
                    y_test = np.concatenate([y_test, j * np.ones(temp_test.shape[0])], axis=0)

            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices, :]
            y_train = y_train[indices]

            # print(x_train.shape)

            # for each factor:
            if x_train.shape[0] == 0 or x_test.shape[0] == 0:
                continue

            model = RandomForestClassifier(n_estimators=10)

            model.fit(x_train, y_train)
            informativeness.append(model.score(x_test, y_test))
            r.append(model.feature_importances_)

            # print(model.feature_importances_.shape) 256

        r = np.array(r)

        for i in range(0, r.shape[1]):
            p = r[:, i]
            p = p / np.sum(p)
            h_k_p = self.entropy(p) / np.log(r.shape[0])
            disentanglement.append(1 - h_k_p)

        disentanglement = np.array(disentanglement)
        weight = np.sum(r, axis=0) / np.sum(r)
        # print("Disentanglement Score: {:.4f}".format(np.sum(weight * disentanglement)))

        for j in range(0, r.shape[0]):
            p = r[j, :]
            p = p / np.sum(p)
            h_d_p = self.entropy(p) / np.log(r.shape[1])
            completeness.append(1 - h_d_p)

        completeness = np.array(completeness)
        # print("Completeness Score: {:.4f}".format(np.mean(completeness)))

        informativeness = np.array(informativeness)
        # print("Informativeness Score: {:.4f}".format(np.mean(informativeness)))

        return {
            # DisentanglementMetric.DISENTANGLEMENT: (np.sum(weight * disentanglement), 0),
            DisentanglementMetric.COMPLETENESS: (np.mean(completeness), np.std(completeness)),
            DisentanglementMetric.INFORMATIVENESS: (np.mean(informativeness), np.std(informativeness))
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
                    y_train, y_test = j * np.ones(temp_train.shape[0]), j * np.ones(temp_test.shape[0])
                else:
                    x_train = np.concatenate([x_train, temp_train], axis=0)
                    x_test = np.concatenate([x_test, temp_test], axis=0)
                    y_train = np.concatenate([y_train, j * np.ones(temp_train.shape[0])], axis=0)
                    y_test = np.concatenate([y_test, j * np.ones(temp_test.shape[0])], axis=0)
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices, :]
            y_train = y_train[indices]

            if (np.sum(y_train) != 0):
                acc = []
                for j in range(0, x_train.shape[1]):
                    temp_x_train, temp_x_test = x_train[:, j].reshape(-1, 1), x_test[:, j].reshape(-1, 1)
                    model = LinearSVC(C=0.01)
                    model.fit(temp_x_train, y_train)
                    acc.append(model.score(temp_x_test, y_test))
                acc.sort(reverse=True)
                sap.append(acc[0] - acc[1])
        # print("SAP score: {:.4f}".format(np.mean(sap)))
        return np.mean(sap)

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


