import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from data.data_utils import GoEmotionsDataset, EmotionsDataset, AppReviewsDataset
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report,confusion_matrix,f1_score,accuracy_score,precision_score,recall_score)
from data.goemotions_enum import GoEmotion, label_mapping
from preprocessing.embedding import load_embedding_model
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
import pandas as pd
import joblib 

def train():
    data_path_goemotions = "C:/Users/sarah/Downloads/data/goEmotions"
    data_path_emotions = "C:/Users/sarah/Downloads/data/Emotions"
    data_path_appreview = "C:/Users/sarah/Downloads/data/AppReviews"
    
    ##################GoEmotions####################

    train_set = GoEmotionsDataset(data_path_goemotions, split="train")
    train_set_labels = [label_mapping.get(x, x) for x in train_set.labels]
    val_set = GoEmotionsDataset(data_path_goemotions, split="val")
    val_set_labels = [label_mapping.get(x, x) for x in val_set.labels]

    ##################Appreview####################

    # train_set=AppReviewsDataset(data_path_appreview,split="train")
    # val_set=AppReviewsDataset(data_path_appreview,split="val")

    #     # --- Train bereinigen ---
    # df_train = pd.DataFrame({
    #     "text": train_set.corpus,
    #     "label": train_set.labels
    # })
    # df_train = df_train.dropna(subset=["text"])
    # df_train = df_train[df_train["text"].str.strip().astype(bool)]
    # train_set_corpus = df_train["text"].tolist()
    # train_set_labels = df_train["label"].tolist()

    # # --- Val bereinigen ---
    # df_val = pd.DataFrame({
    #     "text": val_set.corpus,
    #     "label": val_set.labels
    # })
    # df_val = df_val.dropna(subset=["text"])
    # df_val = df_val[df_val["text"].str.strip().astype(bool)]
    # val_set_corpus = df_val["text"].tolist()
    # val_set_labels = df_val["label"].tolist()

    ##################Emotions####################

    # train_set = EmotionsDataset(data_path_emotions, split="train")
    # val_set = EmotionsDataset(data_path_emotions, split="val")

    #.........................Embedding.........................#

    # vectorizer = TfidfVectorizer(max_features=3000)
    # train_texts=vectorizer.fit_transform(train_set_corpus)
    # val_texts = vectorizer.transform(val_set_corpus)

    # embedding_model = load_embedding_model("fasttext","C:/Users/MEIN PC/Downloads/cc.en.300.bin")
    # train_texts  = texts_to_sentence_embeddings(train_set .corpus, embedding_model,300).numpy()
    # val_texts  = texts_to_sentence_embeddings(val_set .corpus, embedding_model,300).numpy()

    embedding_model = load_embedding_model("glove")
    train_texts  = texts_to_sentence_embeddings(train_set .corpus, embedding_model,50).numpy()
    val_texts  = texts_to_sentence_embeddings(val_set .corpus, embedding_model,50).numpy()

    mlp = MLPClassifier(hidden_layer_sizes=(63,),
                       activation='relu',
                       max_iter=900, 
                       learning_rate_init= 0.00035506214270707734,
                       early_stopping=True,
                       solver='sgd',
                       alpha=0.0013292918943162175,
                       n_iter_no_change=20, 
                       verbose=True)
    mlp.fit(train_texts, train_set_labels)

    print("Training fertig!")
    # joblib.dump(vectorizer, "C:/Users/sarah/Documents/_studium/nlp/modelle/tfdidf/vektorizer_Appreview_tfdidf.pkl")
    # joblib.dump(train_texts, "C:/Users/sarah/Documents/_studium/nlp/modelle/tfdidf/train_texts_Appreview_tfdidf.pkl")
    joblib.dump(mlp, "C:/Users/sarah/Documents/_studium/nlp/modelle/glove/bestModel_GoEmotion_glove.pkl")
    print("Modell gespeichert.")

    # mlp = joblib.load("D:/NLP/modelle_10/mlp_tfdidf_model_emotions.pkl")
    # print("Modell geladen.")
    
    val_preds_emotions = mlp.predict(val_texts)
    accuracy = accuracy_score(val_set_labels, val_preds_emotions)
    print(f"Validation Accuracy: {accuracy:.4f}")

    cm2 = confusion_matrix(val_set_labels, val_preds_emotions)
    class_labels2 = sorted(set(val_set_labels))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm2, annot=True, fmt="d", cmap="YlGn",
            xticklabels=class_labels2,
            yticklabels=class_labels2)
    plt.title("Confusion Matrix - Multiclass MLP")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # Beispiel Vorhersage Zu Testzwecken slaeter loeschen...
    # example = ["I love you."]
    # example_vectorized = vectorizer.transform(example)
    # Y_pred2 = mlp.predict(example_vectorized)
    # print("Vorhersage:", Y_pred2)

    # example = ["I hate you."]
    # example_vectorized = vectorizer.transform(example)
    # Y_pred2 = mlp.predict(example_vectorized)
    # print("Vorhersage:", Y_pred2)

    # example = ["I miss you."]
    # example_vectorized = vectorizer.transform(example)
    # Y_pred2 = mlp.predict(example_vectorized)
    # print("Vorhersage:", Y_pred2)



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

