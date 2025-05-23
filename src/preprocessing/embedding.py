import torch

import gensim.downloader
from preprocessing.tokenization import tokenize_text



class Embedding:

    def __init__(self):
        
        self.glove_wiki_gigaword_50_model = gensim.downloader.load('glove-wiki-gigaword-50')



    def glove_embedding(self, sentence):

        tokens = tokenize_text(sentence)

        emb = self.glove_wiki_gigaword_50_model[tokens]

        return torch.Tensor(emb)

