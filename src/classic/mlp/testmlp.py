import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from data.data_utils import GoEmotionsDataset, EmotionsDataset, AppReviewsDataset
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report,confusion_matrix,f1_score,accuracy_score,precision_score,recall_score)
from data.goemotions_enum import GoEmotion, label_mapping
from preprocessing.embedding import load_embedding_model
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from scipy.stats import randint
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
import joblib
import pandas as pd

def test():
   #data_path = "C:/Users/sarah/Downloads/data/AppReviews"
   
#    train_set = EmotionsDataset(data_path, split="train")
    data_path = "C:/Users/sarah/Downloads/data/goEmotions"
    train_set = GoEmotionsDataset(data_path, split="train")
    train_set_labels = [label_mapping.get(x, x) for x in train_set.labels]
    
    # train_set=AppReviewsDataset(data_path,split="train")
    # val_set=AppReviewsDataset(data_path,split="val")

    n_samples = 20000

    if len(train_set.corpus) > n_samples:
        sampled_corpus, sampled_labels = resample(
            train_set.corpus, train_set_labels, 
            n_samples=n_samples, 
            random_state=42
        )
    else:
        sampled_corpus = train_set.corpus
        sampled_labels = train_set_labels

    # df = pd.DataFrame({
    #     "text": sampled_corpus,
    #     "label": sampled_labels
    # })
    # df = df.dropna(subset=["text"])
    # df = df[df["text"].str.strip().astype(bool)]
    # sampled_corpus = df["text"].tolist()
    # sampled_labels = df["label"].tolist()

    # embedding_model = TfidfVectorizer(max_features=3000)
    # train_texts = embedding_model.fit_transform(sampled_corpus)
    embedding_model = load_embedding_model("glove")
    train_texts = texts_to_sentence_embeddings(sampled_corpus, embedding_model, 50).numpy() 
    #embedding_model = load_embedding_model("fasttext","C:/Users/MEIN PC/Downloads/cc.en.300.bin")
    #train_texts = texts_to_sentence_embeddings(sampled_corpus, embedding_model,300).numpy()


    hidden_layer_choices = [
        tuple(randint.rvs(60, 200) for _ in range(randint.rvs(1, 2)))
        for _ in range(500) 
    ]
    print("Beispiel Hidden Layer Konfigurationen:", hidden_layer_choices[:10])

    param_dist = {
        'hidden_layer_sizes': hidden_layer_choices,
        'activation': ['relu'],
        'solver': ['sgd'],   # 'adam','sgd'
        'alpha': loguniform(1e-4, 1e-1),  
        'learning_rate_init': loguniform(1e-4, 1e-1),  
        'max_iter': [900],
        'early_stopping': [True]
    }

    mlp = MLPClassifier(random_state=42)

    search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_dist,
        n_iter=100,  #Anzahl der Durchläufe!!!!
        cv=5,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    print("Starte breiten RandomizedSearchCV für MLP...")
    search.fit(train_texts, sampled_labels)
    print("Fertig. Beste Parameter:", search.best_params_)

    best_model = search.best_estimator_
    joblib.dump(best_model, "C:/Users/sarah/Documents/_studium/nlp/modelle/bestModel_goemotions_glove.pkl")
   #joblib.dump(embedding_model, "C:/Users/sarah/Documents/_studium/nlp/modelle/embedding_model_appreviews.pkl")
    print("Bestes Modell gespeichert.")

    y_train_pred = best_model.predict(train_texts)
    accuracy = accuracy_score(sampled_labels, y_train_pred)
    print(f"Training Accuracy des besten Modells: {accuracy:.4f}")

#   best_model.predict(embedding_model.transform(["I love you to the moon and back."]))

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