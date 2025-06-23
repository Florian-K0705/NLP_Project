import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import data.data_utils as data_utils
from data.data_utils import GoEmotionsDataset, EmotionsDataset, AppReviewsDataset
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (classification_report,confusion_matrix,f1_score,accuracy_score,precision_score,recall_score)
from data.goemotions_enum import GoEmotion, label_mapping
from preprocessing.embedding import load_embedding_model
from sklearn.utils import resample
from scipy.stats import uniform,loguniform
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import joblib


def test_glove():
    # data_path_emotions = "C:/Users/sarah/Downloads/data/Emotions"

    # train_set = EmotionsDataset(data_path_emotions, split="train")
    # val_set = EmotionsDataset(data_path_emotions, split="val")

    data_path = "C:/Users/sarah/Downloads/data/goEmotions"
    train_set = GoEmotionsDataset(data_path, split="train")
    train_set.labels = [label_mapping.get(x, x) for x in train_set.labels]
    val_set = GoEmotionsDataset(data_path, split="val")
    val_set.labels = [label_mapping.get(x, x) for x in val_set.labels]


    n_samples = 3000
    if len(train_set.corpus) > n_samples:
        sampled_corpus, sampled_labels = resample(
            train_set.corpus, train_set.labels,
            n_samples=n_samples,
            random_state=42,
            stratify=train_set.labels
        )
    else:
        sampled_corpus = train_set.corpus
        sampled_labels = train_set.labels

    df = pd.DataFrame({
        "text": sampled_corpus,
        "label": sampled_labels
    })
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().astype(bool)]
    sampled_corpus = df["text"].tolist()
    sampled_labels = df["label"].tolist()

    # ---- Embeddings
    embedding_model = load_embedding_model("glove")

    X_train = texts_to_sentence_embeddings(sampled_corpus, embedding_model)
    X_val = texts_to_sentence_embeddings(val_set.corpus, embedding_model)

    # ---- Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # ---- Grid Search SVM
    param_grid = {
        'estimator__kernel': ['rbf'],
        'estimator__C': [0.5,1,0.9,0.8,0.7,0.6],
        'estimator__gamma': ['auto']  # ['scale', 'auto']
    }

    base_model = OneVsRestClassifier(
        SVC(probability=True, class_weight='balanced')
    )

    search = GridSearchCV(
        base_model,
        param_grid=param_grid,
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    print("Starte GridSearchCV...")
    search.fit(X_train, sampled_labels)
    print("Fertig. Beste Parameter:", search.best_params_)

    best_model = search.best_estimator_
    scores = cross_val_score(best_model, X_train, sampled_labels, cv=5)
    print("Cross-Validation Accuracy: %.3f +/- %.3f" % (scores.mean(), scores.std()))

    joblib.dump(best_model, "C:/Users/sarah/Documents/_studium/nlp/modelle/svm/best_model_glove_goEmozions_svm.pkl")
    print("Modell gespeichert.")

    # ---- Evaluate
    val_preds_emotions = best_model.predict(X_val)
    accuracy = accuracy_score(val_set.labels, val_preds_emotions)
    print(f"Validation Accuracy: {accuracy:.4f}")

    cm2 = confusion_matrix(val_set.labels, val_preds_emotions)
    class_labels2 = sorted(set(val_set.labels))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm2, annot=True, fmt="d", cmap="YlGn",
                xticklabels=class_labels2,
                yticklabels=class_labels2)
    plt.title("Confusion Matrix - Multiclass SVM")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    val_true = val_set.labels
    val_probs = best_model.predict_proba(X_val)

    # Labels binarize für ROC AUC
    classes = sorted(set(val_true))
    val_true_bin = label_binarize(val_true, classes=classes)

    # Klassische Scores
    precision = precision_score(val_true, val_preds_emotions, average='macro')
    recall = recall_score(val_true, val_preds_emotions, average='macro')
    f1 = f1_score(val_true, val_preds_emotions, average='macro')

    # ROC AUC
    roc_auc = roc_auc_score(val_true_bin, val_probs, average='macro', multi_class='ovr')

    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    print(f"ROC AUC (macro OvR): {roc_auc:.4f}")


def sentence_embedding(text, embedding_model):
    words = text.split()  # oder tokenize_text(text)
    vectors = []
    for w in words:
        try:
            v = embedding_model.glove_wiki_gigaword_50_model[w]
            vectors.append(v)
        except KeyError:
            continue
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_model.glove_wiki_gigaword_50_model.vector_size)


def texts_to_sentence_embeddings(texts, embedding_model):
    return np.array([sentence_embedding(text, embedding_model) for text in texts])


