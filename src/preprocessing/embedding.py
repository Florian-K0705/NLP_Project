import torch

#import gensim.downloader.load as l
from preprocessing.tokenization import tokenize_text



class Embedding:

    def __init__(self, glove_path, embedding_dim=50, max_length=64):
        self.embedding_dim = embedding_dim
        self.word_to_vec = {}
        self.max_length = max_length
        self.SOS = torch.zeros(50)
        self.SOS[49] = 1
        self.EOS = torch.ones(50)
        self.PAD = torch.zeros(50)

        with open(glove_path, 'r', encoding='utf8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
                self.word_to_vec[word] = vec



    def glove_embedding(self, sentence):
        


        tokens = tokenize_text(sentence)

        emb = [self.SOS]
        for index, token in enumerate(tokens):
            if index == self.max_length - 2:
                break
            try:
               emb.append(self.word_to_vec[token])
            except:
               continue
        
        while len(emb) < self.max_length - 1:
            emb.append(self.PAD)

        emb.append(self.EOS)
        return emb


