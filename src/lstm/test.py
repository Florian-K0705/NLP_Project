import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
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
            
            predicted_labels.append(label_mapping[predict.item()])
            real_labels.append(label_mapping[label])

        # === Neu: Metriken berechnen ===
    accuracy = accuracy_score(real_labels, predicted_labels)  # === Neu ===
    f1 = f1_score(real_labels, predicted_labels, average='macro')  # === Neu ===
    recall = recall_score(real_labels, predicted_labels, average='macro')  # === Neu ===

    # Für ROC AUC brauchen wir Wahrscheinlichkeiten und numerische Labels
    # Deshalb berechnen wir hier nochmal die Vorhersagewahrscheinlichkeiten und numerische Labels
    # Wir wandeln Labels in Integer (Klassenindices) um, dazu brauchen wir eine Umkehr-Mapping:
    inv_label_mapping = {v: k for k, v in label_mapping.items()}  # === Neu ===

    y_true_int = np.array([inv_label_mapping[label] for label in real_labels])  # === Neu ===

    # Wahrscheinlichkeiten sammeln
    probs_list = []
    with torch.no_grad():
        for text, _ in tqdm(dataset, desc="Probabilities calc"):  # === Neu ===
            emb = embedder.glove_embedding(text)
            emb = emb.unsqueeze(0)
            outputs = model(emb)
            prob = torch.softmax(outputs, dim=1)
            probs_list.append(prob.squeeze().cpu().numpy())
    y_proba = np.array(probs_list)  # === Neu ===

    try:
        roc_auc = roc_auc_score(y_true_int, y_proba, multi_class='ovr')  # === Neu ===
    except ValueError:
        roc_auc = None  # === Neu ===

    print(f"Accuracy: {accuracy:.4f}")  # === Neu ===
    print(f"F1-Score: {f1:.4f}")       # === Neu ===
    print(f"Recall: {recall:.4f}")     # === Neu ===
    print(f"ROC AUC: {roc_auc if roc_auc is not None else 'nicht definiert'}")  # === Neu ===


    cm = confusion_matrix(real_labels, predicted_labels)
    plt.imshow(cm, cmap="gray")
    plt.show()
    return




label_mapping = {
    0: 0,    # ADMIRATION
    1: 1,    # AMUSEMENT
    2: 2,    # ANGER
    3: 3,    # ANNOYANCE
    4: 4,    # APPROVAL
    5: 5,   # CARING → LOVE #################### Swaped
    6: 3,    # CONFUSION → ANNOYANCE
    7: 7,    # CURIOSITY
    8: 7,    # DESIRE → CURIOSITY
    9: 9,    # DISAPPOINTMENT
    10: 10,  # DISAPPROVAL
    11: 10,  # DISGUST → DISAPPROVAL
    12: 3,   # EMBARRASSMENT → ANNOYANCE
    13: 1,   # EXCITEMENT → AMUSEMENT
    14: 2,   # FEAR → ANGER
    15: 15,  # GRATITUDE
    16: 9,   # GRIEF → DISAPPOINTMENT
    17: 17,  # JOY
    18: 5,  # LOVE #######################Swapped
    19: 7,   # NERVOUSNESS → CURIOSITY
    20: 20,  # OPTIMISM
    21: 0,   # PRIDE → ADMIRATION
    22: 22,  # REALIZATION
    23: 15,  # RELIEF → GRATITUDE
    24: 10,  # REMORSE → DISAPPROVAL
    25: 9,   # SADNESS → DISAPPOINTMENT
    26: 22,  # SURPRISE → REALIZATION
    27: 27   # NEUTRAL
}