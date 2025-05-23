import torch
import pandas as pd
import preprocessing.embedding as embedding

class Dataset(torch.utils.data.Dataset):

    def __init__(self, corpus, labels):
        self.corpus = corpus
        self.labels = labels

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        
        l = 0

        if self.labels[idx] == "clickbait":
            l = 1
        

        return self.corpus[idx], l



train_data = pd.read_csv("/home/florian/Schreibtisch/Clickbaite_dataset/train.csv")
test_data = pd.read_csv("/home/florian/Schreibtisch/Clickbaite_dataset/test.csv")

train_corpus = train_data["title"].to_list()
train_labels = train_data["label"].to_list()


# Embedding the training corpus
train_corpus_embedding = []

print("Start embedding training corpus")
c = 0
d = 0


for x in train_corpus:
    print(c, d)
    try:
        d += 1
        e = embedding.glove_embedding(x)
    except:
        continue

    train_corpus_embedding.append(e)

    c+= 1

test_corpus = test_data["title"].to_list()
test_labels = test_data["label"].to_list()

# Embedding the test corpus
test_corpus_embedding = []

for x in test_corpus:
    try:
        e = embedding.glove_embedding(x)
    except:
        continue


    test_corpus_embedding.append(e)

train_ds = Dataset(train_corpus_embedding, train_labels)
test_ds = Dataset(test_corpus_embedding, test_labels)