import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
#import preprocessing.embedding as Embedder
#import lstm.models as l
#import data.data_utils as d

def train_lstm(model, dataset, embedder, optimizer, batch_size, criterion, num_epochs=10):
    loss_log = []
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        running_loss = 0.0
        count = 0

        pbar = tqdm(torch.randperm(len(dataset)), colour='yellow')

        batch_texts = []
        batch_labels = []

        for idx in pbar:
            text, label = dataset[idx.item()]
            embedded = embedder.glove_embedding(text)
            


            batch_texts.append(torch.stack(embedded))  # [seq_len, emb_dim]
            batch_labels.append(label)

            if len(batch_texts) == batch_size:
                # Padding
                padded_batch = pad_sequence(batch_texts, batch_first=True)  # [B, max_len, emb_dim]
                labels = torch.tensor(batch_labels, dtype=torch.long)
                # Forward
                outputs = model(padded_batch)  # [B, num_classes]
                
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item()
                count += 1

                pbar.set_description(f"Loss: {running_loss / count:.4f}")

                batch_texts = []
                batch_labels = []

        if count > 0:
            loss_log.append(running_loss / count)

    plt.plot(loss_log)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()