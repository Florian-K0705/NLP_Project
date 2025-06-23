import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from data.data_utils import GoEmotionsDataset,  EmotionsDataset,AppReviewsDataset
from sklearn.utils import resample
from data.goemotions_enum import label_mapping
import pandas as pd

def main():

    data_path_goemotions = "C:/Users/sarah/Downloads/data/goEmotions"
    train_set = GoEmotionsDataset(data_path_goemotions, split="train")
    train_set_labels = [label_mapping.get(x, x) for x in train_set.labels]
    val_set = GoEmotionsDataset(data_path_goemotions, split="val")
    val_set_labels = [label_mapping.get(x, x) for x in val_set.labels]

    # data_path_emotions = "C:/Users/sarah/Downloads/data/Emotions"
    # train_set = EmotionsDataset(data_path_emotions, split="train")
    # val_set = EmotionsDataset(data_path_emotions, split="val")

    # data_path_appreview = "C:/Users/sarah/Downloads/data/AppReviews"
    # train_set=AppReviewsDataset(data_path_appreview,split="train")
    # val_set=AppReviewsDataset(data_path_appreview,split="val")


    # df_train = pd.DataFrame({
    #     "text": train_set.corpus,
    #     "label": train_set.labels
    # })
    # df_train = df_train.dropna(subset=["text"])
    # df_train = df_train[df_train["text"].str.strip().astype(bool)]
    # train_set_corpus = df_train["text"].tolist()
    # train_set_labels = df_train["label"].tolist()

    # df_val = pd.DataFrame({
    #     "text": val_set.corpus,
    #     "label": val_set.labels
    # })
    # df_val = df_val.dropna(subset=["text"])
    # df_val = df_val[df_val["text"].str.strip().astype(bool)]
    # val_set_corpus = df_val["text"].tolist()
    # val_set_labels = df_val["label"].tolist()

    n_samples = 8000
    if len(train_set.corpus) > n_samples:
        sampled_corpus, sampled_labels = resample(
            train_set.corpus, train_set_labels,
            n_samples=n_samples,
            random_state=42
        )
    else:
        sampled_corpus = train_set.corpus
        sampled_labels = train_set_labels


    vectorizer = TfidfVectorizer(max_features=3000)
    X_train = vectorizer.fit_transform(sampled_corpus)
    X_val = vectorizer.transform(val_set.corpus)

    # ---------- Modell ----------
    print("[INFO] Training SVM...")
    model = SVC(kernel='rbf', 
                gamma='scale', 
                C=1.24438490026566,
                 probability=True)
    model.fit(X_train, sampled_labels)
    print("[INFO] Training abgeschlossen.")

    # ---------- Speichern ----------
    joblib.dump(model, "C:/Users/sarah/Documents/_studium/nlp/modelle/svm/tfdidf/svm_goemotio_tfidf.pkl")
    joblib.dump(vectorizer, "C:/Users/sarah/Documents/_studium/nlp/modelle/svm/tfdidf/tfidf_vectorizer_goemotio.pkl")
    print("[INFO] Modell & Vectorizer gespeichert.")

    # ---------- Validierung ----------
    val_preds = model.predict(X_val)

    print("[INFO] Confusion Matrix & Report:")
    cm = confusion_matrix(val_set_labels, val_preds)
    class_labels = sorted(set(val_set_labels))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn",
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title("Confusion Matrix - SVM")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    print(classification_report(val_set_labels, val_preds))

    # ---------- Beispiel Vorhersagen ----------
    examples = ["I love you.", "I hate you.", "I miss you."]
    for ex in examples:
        vec = vectorizer.transform([ex])
        pred = model.predict(vec)
        print(f"Beispiel: \"{ex}\" => Vorhersage: {pred}")
