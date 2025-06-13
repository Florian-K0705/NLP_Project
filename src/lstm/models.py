import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim=50, hidden_dim=128, output_dim=28):
        super().__init__()
        self.LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        _, (hidden, _) = self.LSTM(x)
        return self.fc(hidden.squeeze(0))






