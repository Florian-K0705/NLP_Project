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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import joblib


def test():
    data_path_emotions = "C:/Users/sarah/Downloads/data/Emotions"
    #data_path = "C:/Users/sarah/Downloads/data/AppReviews"

    train_set = EmotionsDataset(data_path_emotions, split="train")
    val_set = EmotionsDataset(data_path_emotions, split="val")

    # data_path = "C:/Users/sarah/Downloads/data/goEmotions"
    # train_set = GoEmotionsDataset(data_path, split="train")
    # train_set.labels = [label_mapping.get(x, x) for x in train_set.labels]
    # val_set = GoEmotionsDataset(data_path, split="val")
    # val_set.labels = [label_mapping.get(x, x) for x in val_set.labels]

    # train_set = EmotionsDataset(data_path_emotions, split="train")
    # val_set = EmotionsDataset(data_path_emotions, split="val")

    #train_set=AppReviewsDataset(data_path,split="train")
    #val_set=AppReviewsDataset(data_path,split="val")

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

    # vectorizer = TfidfVectorizer(max_features=2000)
    # train_texts = vectorizer.fit_transform(sampled_corpus)
    # embedding_model = load_embedding_model("glove")
    # train_texts = texts_to_sentence_embeddings(sampled_corpus, embedding_model, 50).numpy() 
    # val_texts = texts_to_sentence_embeddings(val_set.corpus, embedding_model, 50).numpy()
    embedding_model = load_embedding_model("fasttext","C:/Users/sarah/Downloads/cc.en.300.bin")
    train_texts = texts_to_sentence_embeddings(sampled_corpus, embedding_model,300).numpy()
    val_texts = texts_to_sentence_embeddings(val_set.corpus, embedding_model,300).numpy()


    # param_dist = {
    #     'estimator__kernel': ['rbf'],#'rbf','linear'
    #     'estimator__C': loguniform(1e-4, 1e2),#1e-4, 1e2
    #     'estimator__gamma': ['scale', 'auto']#'scale', 'auto'
    # }

    # base_model = OneVsRestClassifier(
    #     SVC(probability=True, class_weight='balanced')
    # )

    # search = RandomizedSearchCV(
    #     base_model,
    #     param_distributions=param_dist,
    #     n_iter=60,
    #     cv=3,
    #     verbose=2,
    #     n_jobs=-1,
    #     random_state=42
    # )

    scaler = StandardScaler()
    train_texts = scaler.fit_transform(train_texts)
    val_texts = scaler.transform(val_texts)

    # ---- Grid Search SVM
    param_grid = {
        'estimator__kernel': ['rbf'],
        'estimator__C': [0.5,1,5,10,0.01],
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

    print("Starte RandomizedSearchCV...")
    search.fit(train_texts, sampled_labels)
    print("Fertig. Beste Parameter:", search.best_params_)

    best_model = search.best_estimator_
    scores = cross_val_score(best_model, train_texts, sampled_labels, cv=3)
    print("Cross-Validation Accuracy: %.3f +/- %.3f" % (scores.mean(), scores.std()))

    joblib.dump(best_model, "C:/Users/sarah/Documents/_studium/nlp/modelle/svm/best_model_fasttext_goEmotions_svm.pkl")
    #joblib.dump(embedding_model, "C:/Users/sarah/Documents/_studium/nlp/modelle/svm/vectorizer_glove_svm.pkl")
    print("Modell & Vektorizer gespeichert.")

    val_preds_emotions = best_model.predict(val_texts)
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
    val_probs = best_model.predict_proba(val_texts)

    # Labels binarize für ROC AUC
    classes = sorted(set(val_true))
    val_true_bin = label_binarize(val_true, classes=classes)

    # Klassische Scores
    precision = precision_score(val_true, val_preds_emotions, average='macro', zero_division=0)
    recall = recall_score(val_true, val_preds_emotions, average='macro', zero_division=0)
    f1 = f1_score(val_true, val_preds_emotions, average='macro', zero_division=0)

    # ROC AUC
    roc_auc = roc_auc_score(val_true_bin, val_probs, average='macro', multi_class='ovr')

    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    print(f"ROC AUC (macro OvR): {roc_auc:.4f}")

def sentence_embedding_mean(embedding_model, zeros, sentence):
    try:
        word_embeddings = embedding_model.get_sentence_embeddings(sentence)
        if word_embeddings is None or len(word_embeddings) == 0:
            return torch.zeros(zeros)  # Kein Wort bekannt
        return word_embeddings.mean(dim=0)
    except Exception as e:
        return torch.zeros(zeros)

def texts_to_sentence_embeddings(texts, embedding_model,zeros):
    return torch.stack([sentence_embedding_mean(embedding_model, zeros,text) for text in texts])

