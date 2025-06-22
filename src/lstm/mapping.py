import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np



def simple_label_mapping(model, embedder, dataset, outputsize):

    label_mapping =  label_mapping1

    """
    Für jedes echte Label i:
    - Sammle alle Texte mit Label i
    - Lass sie als Batch durch das Model laufen
    - Berechne den Durchschnitt der vorhergesagten Klassen (torch.argmax)
    - Drucke das Mapping
    
    Annahme: dataset ist eine Liste von (text, label)
    """
    model.eval()

    for i in range(outputsize-1):
        # Alle Texte mit Label true_label sammeln
        texts_i = [text for text, label in dataset if label == i]

        with torch.no_grad():
            embeddings = [model(embedder.glove_embedding(text)) for text in texts_i]  # Liste von [seq_len, emb_dim]
            batch = torch.stack(embeddings)  # [batch_size, seq_len, emb_dim]

            outputs = torch.stack(embeddings) # [batch_size, output_dim]
            preds = torch.argmax(outputs, dim=1)  # [batch_size]
            avg_pred = preds.float().mean().item()

        print(f"{label_mapping[i]} -> durchschnittliche Modellklasse: {avg_pred:.2f} (gerundet {round(avg_pred)})")



label_mapping1 = {
    0: "ADMIRATION",
    1: "AMUSEMENT",
    2: "ANGER",
    3: "ANNOYANCE",
    4: "APPROVAL",
    5: "OPTIMISM",       # CARING → OPTIMISM
    6: "CURIOSITY",      # CONFUSION → CURIOSITY
    7: "CURIOSITY",
    8: "OPTIMISM",       # DESIRE → OPTIMISM
    9: "DISAPPOINTMENT",
    10: "DISAPPROVAL",
    11: "ANGER",         # DISGUST → ANGER
    12: "DISAPPOINTMENT",# EMBARRASSMENT → DISAPPOINTMENT
    13: "JOY",           # EXCITEMENT → JOY
    14: "SADNESS",       # FEAR → SADNESS
    15: "GRATITUDE",
    16: "SADNESS",       # GRIEF → SADNESS
    17: "JOY",
    18: "LOVE",
    19: "SADNESS",       # NERVOUSNESS → SADNESS
    20: "OPTIMISM",
    21: "ADMIRATION",    # PRIDE → ADMIRATION
    22: "REALIZATION",
    23: "GRATITUDE",     # RELIEF → GRATITUDE
    24: "SADNESS",       # REMORSE → SADNESS
    25: "SADNESS",
    26: "CURIOSITY",     # SURPRISE → CURIOSITY
    27: "NEUTRAL"
}

label_mapping2 = {
0: 0,
1: 1,
2: 2,
3: 3,
4: 4,
5:5
}