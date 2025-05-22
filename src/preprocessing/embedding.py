import torch

import gensim.downloader
from preprocessing.tokenization import tokenize_text




def glove_embedding(sentence):

    tokens = tokenize_text(sentence)

    embed_model = gensim.downloader.load('glove-wiki-gigaword-50')
    emb = embed_model[tokens]

    return torch.Tensor(emb)

