from collections import OrderedDict
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        bow_dict = {}
        for elem in X:
            lst = elem.split()
            for word in lst:
                if word in bow_dict:
                    bow_dict[word] += 1
                else:
                    bow_dict[word] = 1
        sorted_bow = {k: v for k, v in sorted(bow_dict.items(), key=lambda item: -item[1])}
        self.bow = [0] * self.k
        for i, d in enumerate(sorted_bow):
            if i == self.k:
                break
            self.bow[i] = d
        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """
        result = [0] * self.k
        lst = text.split()
        for word in lst:
            if word in self.bow:
                ind = self.bow.index(word)
                result[ind] += 1
            else:
                continue
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize
        self.freq_lst = [0] * self.k
        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        bow_dict = {}
        term_in_doc = {}
        for elem in X:
            lst = elem.split()
            unique_token = set(lst)
            for word in lst:
                if word in bow_dict:
                    bow_dict[word] += 1
                else:
                    bow_dict[word] = 1
            for word in unique_token:
                if word in term_in_doc:
                    term_in_doc[word] += 1
                else:
                    term_in_doc[word] = 1
                
        sorted_bow = {k: v for k, v in sorted(bow_dict.items(), key=lambda item: -item[1])}
        for i, d in enumerate(sorted_bow):
            if i == self.k:
                break
            self.freq_lst[i] = d
        for elem in self.freq_lst:
            self.idf[elem] = np.log(term_in_doc[elem] / len(X))
        
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """
        result = [0] * self.k
        lst = text.split()
        for word in lst:
            if word in self.idf:
                ind = self.freq_lst.index(word)
                result[ind] += 1 / len(lst)
            else:
                continue
        for i in range(self.k):
            result[i] = result[i] * list(self.idf.items())[i][1]
        if self.normalize:
            x = np.array(result, "float32")
            if np.linalg.norm(x) == 0:
                return x
            return x / np.linalg.norm(x)
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])
    
    def get_vocabulary(self) -> Union[List[str], None]:
        return self.freq_lst
    
    def get_idf(self) -> Union[List[str], None]:
        return self.idf
