import torch
from nltk.tokenize import word_tokenize
import gensim.downloader

import data.data_utils as data_utils
import preprocessing.embedding as embedding
from models import GoEmotionsBasicTransformerModel

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix



def embedding(sentence):

    tokens = word_tokenize(sentence.lower())

    embed_model = gensim.downloader.load('glove-wiki-gigaword-50')
    emb = embed_model[tokens]

    return torch.Tensor(emb)






def train(model, dataset, optimizer, criterion, device, num_classes, num_epochs=10):

    oneHot = torch.eye(num_classes)
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        for text, label in tqdm(dataset):
            

            try:
                embedded_text = embedding(text).unsqueeze(dim=0).to(device)
            except:
                continue


            outputs = model(embedded_text)
            loss = criterion(outputs, oneHot[label].unsqueeze(dim=0).to(device))

            loss.backward()



        optimizer.step()
        print("loss", loss.item())


def test(model, dataset, device):

    model.eval()
    model.to(device)

    predicted_labels = []

    with torch.no_grad():
        for embedded_text, label in tqdm(dataset):

            output = model(embedded_text.unsqueeze(dim=0).to(device))
            _, predicted = torch.max(output.data, 1)

            predicted_labels.append(predicted.item())

    print("Accuracy: ", accuracy_score(dataset.labels, predicted_labels))
    print("Confusion Matrix: ", confusion_matrix(dataset.labels, predicted_labels))



    



def main():

    model = GoEmotionsBasicTransformerModel(num_layers=1, num_heads=1, feature_dim=50, num_classes=2)


    """
    # Start training    
    train(model, train_ds, 
          optimizer=torch.optim.Adam(model.parameters(), lr=0.001), 
          criterion=torch.nn.CrossEntropyLoss(), 
          device="cuda", 
          num_classes=2, 
          num_epochs=1)
    
    # Start testing
    test(model, test_ds, device="cuda")
    """