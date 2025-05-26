import torch
from preprocessing.embedding import Embedding
from tqdm import tqdm
import matplotlib.pyplot as plt





def train(model, dataset, optimizer, batch_size, criterion, device, num_classes, num_epochs=10):

    oneHot = torch.eye(num_classes)
    loss_log = []

    model.train()
    model.to(device)

    c = 0


    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        
        running_loss = 0.0

        

        pbar = tqdm(torch.randperm(len(dataset)), colour='yellow')
        
        for i in pbar:

            if i.item() % batch_size == 0 and i.item() != 0:
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_description(f"Loss: {running_loss / batch_size:.4f}")

                loss_log.append(running_loss / c)
                running_loss = 0.0
                c = 0

            text = dataset[i.item()][0]
            label = dataset[i.item()][1]


            outputs = model(text, device=device)
            loss = criterion(outputs, oneHot[label].unsqueeze(dim=0).to(device))

            loss.backward()
            c += 1

            running_loss += loss.item()


        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description(f"Loss: {running_loss / batch_size:.4f}")
        loss_log.append(running_loss / c)
        c = 0



          
    plt.plot(loss_log)
    plt.show()