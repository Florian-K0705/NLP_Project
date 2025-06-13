import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import logging

def test(model, embedder, dataset):
    model.eval()

    predicted_labels = []
    real_labels = []


    with torch.no_grad():
        for text,label in tqdm(dataset):

            emb = embedder.glove_embedding(text)
            
            emb = emb.unsqueeze(0)
            predd = model(emb)
            predict = torch.argmax(predd)

            predicted_labels.append(predict.item())
            real_labels.append(label)


        cm = confusion_matrix(real_labels, predicted_labels)
        plt.imshow(cm, cmap="gray")
        plt.show()
    return




