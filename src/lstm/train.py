import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_lstm(model, dataset, embedder, optimizer, criterion, batch_size, num_epochs, pth_path):
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
            batch_labels.append(label)

            embedded = embedder.glove_embedding(text) # torch.shape([64, 50]) ([seq_len, emb_dim])
            batch_texts.append(embedded)  
            
            #Batch ready, do what has to be done
            if len(batch_texts) == batch_size:
                
                batch_tensor = torch.stack(batch_texts) # torch.shape([ batch_size, seq_len, emb_dim])
                label_tensor = torch.tensor(batch_labels, dtype=torch.long)

                optimizer.zero_grad()

                outputs = model(batch_tensor)  # torch.shape( [batch_size, output_dim ])
                
                loss = criterion(outputs, label_tensor)
                loss.backward()
                optimizer.step()


                running_loss += loss.item()
                count += 1

                #bereie nächste Batch vor:
                batch_texts = []
                batch_labels = []
            #Batch "vorbei"
                


        tqdm.write(f"Epoch {epoch+1}, Loss: {running_loss / count:.4f}")
        torch.save(model.state_dict(), pth_path)

        loss_log.append(running_loss / count)
       
    
    plt.plot(loss_log)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()