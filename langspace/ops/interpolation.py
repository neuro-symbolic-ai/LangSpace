from typing import List
from torch import Tensor
from gensim.models.keyedvectors import KeyedVectors


class InterpolationOps:
    """
    Operations for obtaining and evaluating interpolations from a source (start) and target (end) representation vectors.

    The linear interpolation method helps in visualizing transitions in latent spaces, which is common in the study
    of generative models. The text methods leverage the Word Mover's Distance (WMD) to evaluate how uniformly transitions
    occur in semantic space, with WMD originally proposed to capture semantic dissimilarity between texts.
    """
    @staticmethod
    def linearize_interpolate(source: Tensor, target: Tensor, size: int = 10) -> List[Tensor]:
        """
        Performs linear interpolation between two representation vectors.

        This method generates a sequence of vectors transitioning from the source to the target by computing the weighted
        average of the two. The interpolation is performed in equal increments, with the weights for the source vector
        decreasing from 1 to 0 and those for the target vector increasing from 0 to 1 over the specified number of steps.

        Args:
            source (Tensor): The starting representation vector.
            target (Tensor): The ending representation vector.
            size (int, optional): The number of interpolation steps between the source and target. Default is 10.

        Returns:
            A list of interpolated vectors, including both the source and target, ordered sequentially.
        """
        return [source * (1-i/size) + target * i/size for i in range(size+1)]

    @staticmethod
    def preprocess(sentence: str, stop_words: List[str]) -> List[str]:
        """
        Normalizes and tokenizes a sentence by lowercasing and removing stop words.

        This method splits the sentence into words after converting it to lowercase and filters out any words that are
        present in the provided stop words list.

        Args:
            sentence (str): The input sentence to preprocess.
            stop_words (List[str]): A list of stop words to exclude from the tokenized output.

        Returns:
            A list of processed words with stop words removed.
        """
        return [w for w in sentence.lower().split() if w not in stop_words]

    @staticmethod
    def word_mover_distance(sent1: str, sent2: str, model: KeyedVectors, stopword: List[str]) -> float:
        """
        Calculates the Word Mover's Distance (WMD) between two sentences.

        This method first preprocesses the input sentences to remove stop words and normalize the text, and then computes the
        WMD between them using the provided word embedding model. WMD reflects the minimum cumulative distance required to
        'move' the embeddings of words in one sentence to match those of the other sentence, thereby capturing semantic
        dissimilarities.

        Args:
            sent1 (str): The first sentence.
            sent2 (str): The second sentence.
            model (KeyedVectors): A word embedding model that supports computing the WMD.
            stopword (List[str]): A list of stop words to remove during preprocessing.

        Returns:
            The computed Word Mover's Distance representing the semantic difference between the two sentences.
        """
        sent1 = InterpolationOps.preprocess(sent1, stopword)
        sent2 = InterpolationOps.preprocess(sent2, stopword)
        distance = model.wmdistance(sent1, sent2)
        return distance

    @staticmethod
    def interpolation_smoothness(interpolate_path: List[str], model_wmd: KeyedVectors, stop_words: List[str]) -> float:
        """
        Calculates the smoothness of an interpolated path between sentences based on Word Mover's Distance.

        This method computes a smoothness score for a sequence of sentences that represent a semantic interpolation path.
        The overall semantic distance (d_origin) is measured between the first and the last sentence of the path.
        Additionally, the cumulative distance between consecutive sentence pairs is computed. The smoothness score is then
        defined as the ratio of the overall distance to the sum of local distances. A score closer to 1 indicates that the
        transition between each adjacent pair of sentences is uniformly distributed, suggesting a smooth semantic change.

        Args:
            interpolate_path (List[str]): A list of sentences forming the interpolation path.
            model_wmd (KeyedVectors): The word embedding model used to compute the Word Mover's Distance.
            stop_words (List[str]): A list of stop words to be removed during the preprocessing of sentences.

        Returns:
            float: The computed smoothness score of the interpolation path. Values closer to 1 imply smoother transitions.

        Evaluating interpolation smoothness using word-level transport distances leverages ideas from metric learning in text
        representations .
        """
        source, target = interpolate_path[0], interpolate_path[-1]
        d_origin = InterpolationOps.word_mover_distance(source, target, model_wmd, stop_words)
        list_d = []
        for j in range(len(interpolate_path)-1):
            d = InterpolationOps.word_mover_distance(interpolate_path[j], interpolate_path[j+1], model_wmd, stop_words)
            list_d.append(d)

        return d_origin / sum(list_d)
