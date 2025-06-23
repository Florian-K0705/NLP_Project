import torch
from data.data_utils import GoEmotionsDataset, EmotionsDataset, AppReviewsDataset
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (classification_report,confusion_matrix,f1_score,accuracy_score,precision_score,recall_score)
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
import pandas as pd
import joblib
from collections import defaultdict
import classic.emotions_to_star as mapping
from sklearn.preprocessing import label_binarize


def emotions_evaluation():

    #data_path_emotions = "C:/Users/sarah/Downloads/data/Emotions"
    data_path_emotions = "C:/Users/sarah/Downloads/data/goEmotions"
    data_path_appreview = "C:/Users/sarah/Downloads/data/AppReviews"
    #test_set_emotions = EmotionsDataset(data_path_emotions, split="train")
    val_set_app =AppReviewsDataset(data_path_appreview,split="val")
    test_set_emotions = GoEmotionsDataset(data_path_emotions, split="train")

    
    mlp_model = joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/mlp/tfdidf/goEmotions/bestModel_goemotion_tfdidf.pkl")
    mlp_embedding= joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/mlp/tfdidf/goEmotions/vektorizer_goemotion_tfdidf.pkl")
    svm_model=joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/svm/tfdidf/svm_goemotio_tfidf.pkl")
    svm_embedding= joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/svm/tfdidf/tfidf_vectorizer_goemotio.pkl")

    #Daten bereinignen
    val_set_app.corpus = ["" if pd.isna(doc) else str(doc) for doc in val_set_app.corpus]
    

    X_test_emotions_mlp = mlp_embedding.transform(test_set_emotions.corpus)
    X_test_emotions_svm = svm_embedding.transform(test_set_emotions.corpus)

    X_test_app_mlp = mlp_embedding.transform(val_set_app.corpus)
    X_test_app_svm = svm_embedding.transform(val_set_app.corpus)

    # === 4) Evaluiere Emotions-Testset ===
    print("\n=== [MLP] Emotions-Testset ===")
    preds_mlp = mlp_model.predict(X_test_emotions_mlp)
    acc_mlp = accuracy_score(test_set_emotions.labels, preds_mlp)
    print(f"Accuracy: {acc_mlp:.4f}")
    print(classification_report(test_set_emotions.labels, preds_mlp))
    cm_mlp = confusion_matrix(test_set_emotions.labels, preds_mlp)
    sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="YlGn")
    plt.title("MLP Confusion Matrix - Emotions Test")
    plt.show()

    print("\n=== [SVM] Emotions-Testset ===")
    preds_svm = svm_model.predict(X_test_emotions_svm)
    acc_svm = accuracy_score(test_set_emotions.labels, preds_svm)
    print(f"Accuracy: {acc_svm:.4f}")
    print(classification_report(test_set_emotions.labels, preds_svm))
    cm_svm = confusion_matrix(test_set_emotions.labels, preds_svm)
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="YlOrRd")
    plt.title("SVM Confusion Matrix - Emotions Test")
    plt.show()

    # === 5) Mapping: AppReviews → Emotions ===
    print("\n=== [Mapping: AppReviews → Emotions-Vorhersage] ===")

    preds_app_mlp = mlp_model.predict(X_test_app_mlp)
    preds_app_svm = svm_model.predict(X_test_app_svm)

    # Mapping zählen
    mapping_mlp = defaultdict(list)
    for app_label, emotion_label in zip(val_set_app.labels, preds_app_mlp):
        mapping_mlp[app_label].append(emotion_label)

    mapping_svm = defaultdict(list)
    for app_label, emotion_label in zip(val_set_app.labels, preds_app_svm):
        mapping_svm[app_label].append(emotion_label)

    # Zähltabelle bauen
    def build_count_table(mapping):
        rows = []
        for app_label, emotion_list in mapping.items():
            counts = pd.Series(emotion_list).value_counts().to_dict()
            row = {"AppLabel": app_label}
            row.update(counts)
            rows.append(row)
        return pd.DataFrame(rows).fillna(0).set_index("AppLabel").astype(int)

    df_mlp = build_count_table(mapping_mlp)
    df_svm = build_count_table(mapping_svm)
    #für ausgewogenere Ergenisse als test...
    df_mlp_norm = df_mlp.div(df_mlp.sum(axis=1), axis=0)
    df_svm_norm = df_svm.div(df_svm.sum(axis=1), axis=0)

    print("\n[MLP] Mapping Table:")
    print(df_mlp)

    print("\n[SVM] Mapping Table:")
    print(df_svm)

    plt.figure(figsize=(12, 8))
    sns.heatmap(df_mlp_norm, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("MLP: AppReviews → Emotions Mapping (normalized)")
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df_svm_norm, annot=True, cmap="YlOrRd", fmt=".2f")
    plt.title("SVM: AppReviews → Emotions Mapping (normalized)")
    plt.show()

def evaluate_appreview_models():
    # === Lade Daten und Modelle ===
    data_path_appreview = "C:/Users/sarah/Downloads/data/AppReviews"
    test_set_app = AppReviewsDataset(data_path_appreview, split="test")

    mlp_model = joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/mlp/glove/AppReview/bestModel_Appreview_glove.pkl")
    mlp_embedding = joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/mlp/glove/AppReview/vektorizer_Appreview_glove.pkl")

    svm_model = joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/mlp/tfdidf/appreview/bestModel_Appreview_tfdidf.pkl")
    svm_embedding = joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/mlp/tfdidf/appreview/vektorizer_Appreview_tfdidf.pkl")

    X_test_app_mlp = mlp_embedding.transform(test_set_app.corpus)
    X_test_app_svm = svm_embedding.transform(test_set_app.corpus)

    true_stars = test_set_app.labels
    star_classes = sorted(np.unique(true_stars))
    true_binarized = label_binarize(true_stars, classes=star_classes)

    results = {}

    # === MLP Evaluation ===
    print("\n[MLP Evaluation]")
    preds_emotions_mlp = mlp_model.predict(X_test_app_mlp)
    preds_stars_mlp = [mapping.emotion_to_star_e_mlp.get(e, 3) for e in preds_emotions_mlp]

    acc = accuracy_score(true_stars, preds_stars_mlp)
    f1 = f1_score(true_stars, preds_stars_mlp, average='weighted')
    recall = recall_score(true_stars, preds_stars_mlp, average='weighted')

    try:
        probs_emotions_mlp = mlp_model.predict_proba(X_test_app_mlp)
        probs_stars_mlp = np.zeros((len(test_set_app), len(star_classes)))

        class_list = mlp_model.classes_.tolist()

        for emo_class, star in mapping.emotion_to_star_e_mlp.items():
            if star in star_classes and emo_class in class_list:
                emo_idx = class_list.index(emo_class)
                star_idx = star_classes.index(star)
                probs_stars_mlp[:, star_idx] += probs_emotions_mlp[:, emo_idx]

        roc_auc = roc_auc_score(true_binarized, probs_stars_mlp, multi_class='ovr')
    except Exception as e:
        roc_auc = f"Nicht berechenbar: {e}"

    results['MLP'] = {'accuracy': acc, 'f1': f1, 'recall': recall, 'roc_auc': roc_auc}

    # === SVM Evaluation ===
    print("\n[SVM Evaluation]")
    preds_emotions_svm = svm_model.predict(X_test_app_svm)
    preds_stars_svm = [mapping.emotion_to_star_e_svm.get(e, 3) for e in preds_emotions_svm]

    acc = accuracy_score(true_stars, preds_stars_svm)
    f1 = f1_score(true_stars, preds_stars_svm, average='weighted')
    recall = recall_score(true_stars, preds_stars_svm, average='weighted')

    try:
        if hasattr(svm_model, "predict_proba"):
            probs_emotions_svm = svm_model.predict_proba(X_test_app_svm)
            probs_stars_svm = np.zeros((len(test_set_app), len(star_classes)))

            class_list = svm_model.classes_.tolist()

            for emo_class, star in mapping.emotion_to_star_e_svm.items():
                if star in star_classes and emo_class in class_list:
                    emo_idx = class_list.index(emo_class)
                    star_idx = star_classes.index(star)
                    probs_stars_svm[:, star_idx] += probs_emotions_svm[:, emo_idx]

            roc_auc = roc_auc_score(true_binarized, probs_stars_svm, multi_class='ovr')
        else:
            roc_auc = "predict_proba() nicht verfügbar für SVM"
    except Exception as e:
        roc_auc = f"Nicht berechenbar: {e}"

    results['SVM'] = {'accuracy': acc, 'f1': f1, 'recall': recall, 'roc_auc': roc_auc}

    # === Ausgabe der Ergebnisse ===
    for model_name, scores in results.items():
        print(f"\n[{model_name}]")
        for metric, value in scores.items():
            print(f"{metric.capitalize()}: {value}")

    return results




def test_appreview():
    data_path_appreview = "C:/Users/sarah/Downloads/data/AppReviews"
    test_set_app =AppReviewsDataset(data_path_appreview,split="val")
    
    mlp_model = joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/mlp/tfdidf/emotions/bestModel_emotion_tfdidf.pkl")
    mlp_embedding= joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/mlp/tfdidf/emotions/vektorizer_emotion_tfdidf.pkl")
    svm_model=joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/svm/tfdidf/svm_emotions_tfidf.pkl")
    svm_embedding= joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/svm/tfdidf/tfidf_vectorizer_emotions.pkl")
    # mlp_model = joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/mlp/tfdidf/goEmotions/bestModel_goemotion_tfdidf.pkl")
    # mlp_embedding= joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/mlp/tfdidf/goEmotions/vektorizer_goemotion_tfdidf.pkl")
    # svm_model=joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/svm/tfdidf/svm_goemotio_tfidf.pkl")
    # svm_embedding= joblib.load("C:/Users/sarah/Documents/_studium/nlp/modelle/svm/tfdidf/tfidf_vectorizer_goemotio.pkl")


    test_set_app=AppReviewsDataset(data_path_appreview, split="test")

    X_test_app_mlp = mlp_embedding.transform(test_set_app.corpus)
    X_test_app_svm = svm_embedding.transform(test_set_app.corpus)

    preds_app_mlp = mlp_model.predict(X_test_app_mlp)
    preds_app_svm = svm_model.predict(X_test_app_svm)

    stars_app_mlp = map_emotions_to_stars(preds_app_mlp, mapping.emotion_to_star_e_mlp)
    stars_app_svm = map_emotions_to_stars(preds_app_svm, mapping.emotion_to_star_e_svm)

    print("\nMLP predicted stars:", stars_app_mlp)
    print("SVM predicted stars:", stars_app_svm)

    true_stars = test_set_app.labels
    print("True stars:", true_stars)

    acc_mlp = accuracy_score(true_stars, stars_app_mlp)
    acc_svm = accuracy_score(true_stars, stars_app_svm)

    f1_mlp = f1_score(true_stars, stars_app_mlp, average="weighted")
    f1_svm = f1_score(true_stars, stars_app_svm, average="weighted")

    recall_mlp = recall_score(true_stars, stars_app_mlp, average="weighted")
    recall_svm = recall_score(true_stars, stars_app_svm, average="weighted")

    true_stars = test_set_app.labels
    star_classes = sorted(np.unique(true_stars))  # z.B. [1, 2, 3, 4, 5]

    # Binarisierung
    true_binarized = label_binarize(true_stars, classes=star_classes)

    # Wahrscheinlichkeiten (aus predict_proba)
    probs_emotions_mlp = mlp_model.predict_proba(X_test_app_mlp)

    # Mapping der Wahrscheinlichkeiten von Emotionen zu Sternen
    probs_stars_mlp = np.zeros((len(test_set_app), len(star_classes)))
    for emo_class, star in mapping.emotion_to_star_e_mlp.items():
        if star in star_classes:
            idx = star_classes.index(star)
            probs_stars_mlp[:, idx] += probs_emotions_mlp[:, emo_class]

    if hasattr(svm_model, "predict_proba"):
        probs_emotions_svm = svm_model.predict_proba(X_test_app_svm)
        probs_stars_svm = np.zeros((len(test_set_app), len(star_classes)))
        for emo_class, star in mapping.emotion_to_star_e_svm.items():
            if star in star_classes:
                idx = star_classes.index(star)
                probs_stars_svm[:, idx] += probs_emotions_svm[:, emo_class]
        roc_auc_svm = roc_auc_score(true_binarized, probs_stars_svm, multi_class='ovr')
    else:
        roc_auc_svm = "predict_proba nicht verfügbar"

    # ROC AUC
    roc_auc_mlp = roc_auc_score(true_binarized, probs_stars_mlp, multi_class='ovr')
    print(f"MLP ROC AUC: {roc_auc_mlp:.4f}")
    print(f"\n[MLP] Accuracy: {acc_mlp:.4f}, F1: {f1_mlp:.4f}, Recall: {recall_mlp:.4f}, ROC AUC: {roc_auc_mlp}")
    print(f"[SVM] Accuracy: {acc_svm:.4f}, F1: {f1_svm:.4f}, Recall: {recall_svm:.4f}, ROC AUC: {roc_auc_svm}")

def map_emotions_to_stars(emotion_preds, mapping):
        return [mapping.get(emotion, 3) for emotion in emotion_preds]

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