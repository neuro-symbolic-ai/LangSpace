from typing import List
from torch import Tensor
from gensim.models.keyedvectors import KeyedVectors

class InterpolationOps:
    @staticmethod
    def linearize_interpolate(source: Tensor, target: Tensor, size: int = 10) -> List[Tensor]:
        return [source * (1-i/size) + target * i/size for i in range(size+1)]

    @staticmethod
    def preprocess(sentence: str, stop_words: List[str]) -> List[str]:
        return [w for w in sentence.lower().split() if w not in stop_words]

    @staticmethod
    def word_mover_distance(sent1: str, sent2: str, model: KeyedVectors, stopword: List[str]) -> float:
        sent1 = InterpolationOps.preprocess(sent1, stopword)
        sent2 = InterpolationOps.preprocess(sent2, stopword)
        distance = model.wmdistance(sent1, sent2)
        return distance

    @staticmethod
    def interpolation_smoothness(interpolate_path: List[str], model_wmd: KeyedVectors, stop_words: List[str]) -> float:
        """
        args: list of sentences
        return: value
        """
        source, target = interpolate_path[0], interpolate_path[-1]
        d_origin = InterpolationOps.word_mover_distance(source, target, model_wmd, stop_words)
        list_d = []
        for j in range(len(interpolate_path)-1):
            d = InterpolationOps.word_mover_distance(interpolate_path[j], interpolate_path[j+1], model_wmd, stop_words)
            list_d.append(d)

        return d_origin / sum(list_d)
