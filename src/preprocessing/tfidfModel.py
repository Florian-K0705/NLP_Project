from typing import List
import os
import re
import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

class TfIdfModel:

    def build_index(self, docs: List[List[str]]) -> None:
        all_tokens = set(token for doc in docs for token in doc)
        self.index = {token: idx for idx, token in enumerate(sorted(all_tokens))}

    def train(self, docs: List[List[str]]) -> None:
        self.build_index(docs)
        n_docs = len(docs)
        n_words = len(self.index)

        term_doc_matrix = np.zeros((n_docs, n_words))
        for i, doc in enumerate(docs):
            counts = Counter(doc)
            for word, count in counts.items():
                if word in self.index:
                    term_doc_matrix[i, self.index[word]] = count

        tf_log = np.log1p(term_doc_matrix)
        df_vector = np.count_nonzero(term_doc_matrix, axis=0)
        idf = np.log10(n_docs / (1 + df_vector))
        self.tfidf_matrix = tf_log * idf

    def embed(self, word: str) -> np.ndarray:
        if word in self.index:
            return self.tfidf_matrix[:, self.index[word]]
        return None

    def doc_vector(self, doc: List[str]) -> np.ndarray:
        vec = np.zeros(self.tfidf_matrix.shape[1])
        counts = Counter(doc)
        for word, count in counts.items():
            if word in self.index:
                vec[self.index[word]] = count
        tf_log = np.log1p(vec)
        df_vector = np.count_nonzero(self.tfidf_matrix > 0, axis=0)
        idf = np.log10(self.tfidf_matrix.shape[0] / (1 + df_vector))
        return tf_log * idf