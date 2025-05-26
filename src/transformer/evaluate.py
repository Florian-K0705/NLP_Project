from preprocessing.embedding import Embedding
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def test(model, dataset, device):

    model.eval()
    model.to(device)

    predicted_labels = []
    real_labels = []

    with torch.no_grad():
        for text, label in tqdm(dataset):


            output = model(text, device=device)
            predicted = torch.argmax(output, dim=1)

            predicted_labels.append(predicted.item())
            real_labels.append(label)

    cm = confusion_matrix(real_labels, predicted_labels)
    print("Accuracy: ", accuracy_score(real_labels, predicted_labels))


    plt.imshow(cm, cmap="gray")
    plt.show()