import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import preprocessing.embedding as Embedder
import lstm as lstm
import lstm.models as l

import lstm.train as lt
import data.data_utils as d

data_path = r"C:\Users\bberg\Documents\Natural Language Processing Data\goEmotions"
glov_path = r"C:\Users\bberg\Documents\Natural Language Processing Data\glove.6B.50d.txt"
pth_path = r"C:\Users\bberg\Documents\Natural Language Processing Data\model.pth"

def run():
    train_ds =  d.GoEmotionsDataset(data_path, split="train")
    test_ds = d.GoEmotionsDataset(data_path, split="test")

    max_length = 64#max length of sentence
    batch_size = 4096
    num_epochs = 6

    embedder = Embedder.Embedding(glov_path, 50, max_length)
    model = l.LSTMClassifier(embedding_dim=50)

    try:
        model.load_state_dict(torch.load(pth_path, weights_only=True))
    except:
        pass

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    lt.train_lstm(model, train_ds, embedder, optimizer, batch_size, criterion, num_epochs)


    torch.save(model.state_dict(), pth_path)


