import torch
import torch.nn as nn
import preprocessing.embedding as Embedder

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim=50, hidden_dim=128, output_dim=28):
        super().__init__()

        glov_path = r"C:\Users\bberg\Documents\Natural Language Processing Data\glove.6B.50d.txt"
        #self.embedding = Embedder(glov_path, 50)
        self.LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #embedded = self.embedding(x)
        _, (hidden, _) = self.LSTM(x)
        return self.fc(hidden.squeeze(0))






