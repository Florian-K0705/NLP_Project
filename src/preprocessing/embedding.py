import torch
import numpy as np

import gensim.downloader
from preprocessing.tokenization import tokenize_text
import fasttext
import os


def load_embedding_model(model_name, path=None):
    if model_name == "glove":
        return GloVeEmbeddingModel()
    elif model_name == "fasttext":
        return FastTextEmbeddingModel(path)
    else:
        raise ValueError(f"Unknown embedding model: {model_name}")


class EmbeddingModel():

    def __init__(self):
        pass

    def __call__(self, sentence):
        return self.get_sentence_embeddings(sentence)

    # Diese Methode gibt nur die Embeddings der einzelnen Wörter als "Liste" von Embeddingvektoren zurück (als Pytorch Tensor)
    def get_sentence_embeddings(self, sentence):
        pass


class GloVeEmbeddingModel(EmbeddingModel):
    def __init__(self):
        super().__init__()
        self.glove_wiki_gigaword_50_model = gensim.downloader.load('glove-wiki-gigaword-50')

    def get_sentence_embeddings(self, sentence):
        tokens = tokenize_text(sentence)
        emb = self.glove_wiki_gigaword_50_model[tokens]
        return torch.Tensor(emb)
    

class FastTextEmbeddingModel(EmbeddingModel):
    def __init__(self, path):
        super().__init__()

        self.ft = fasttext.load_model(os.path.join(path, "cc.en.300.bin"))


    def get_sentence_embeddings(self, sentence):
        tokens = tokenize_text(sentence)
        emb = [self.ft.get_word_vector(token) for token in tokens]
        
        embeddings = np.stack(emb, axis=0)

        return torch.Tensor(embeddings)



#################################################
## Das soll refactored werden!

class Embedding:

    def __init__(self):
        
        self.glove_wiki_gigaword_50_model = gensim.downloader.load('glove-wiki-gigaword-50')



    def glove_embedding(self, sentence):

        tokens = tokenize_text(sentence)

        emb = self.glove_wiki_gigaword_50_model[tokens]

        return torch.Tensor(emb)

