#übergangslösung...
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data.data_utils import GoEmotionsDataset
import classic.svm.runLinearSVM as runLinearSVM
import classic.mlp.runMLP as runMLP
import preprocessing.tokenizer as tokenizer
import preprocessing.tfidfModel
import ast
import numpy as np
import pandas as pd
import data.goemotions_enum as Emotions_Enum

data_path = "C:/Users/MEIN PC/Downloads/data/goEmotions"
    # Laden der Daten
train_set = GoEmotionsDataset(data_path, split="train")
test_set = GoEmotionsDataset(data_path, split="test")

    #Teste erstmal nur mit den ersten 10 Sätzen
train_set = [train_set[i] for i in range(2000)]
test_set = [test_set[i] for i in range(1000)]

# train_set = [
#      ("Ich bin so enttäuscht von dir.", -1),
#      ("Das war der schlimmste Tag meines Lebens.", -1),
#      ("Niemand hört mir je zu.", -1),
#      ("Ich hasse es, wenn Leute so unfreundlich sind.", -1),
#      ("Mir geht es richtig schlecht heute.", -1),
#      ("Ich liebe es, neue Dinge zu lernen.", 0),
#      ("Ich fühle mich heute richtig gut.", 0),
#      ("Das Abendessen gestern war großartig!", 0),
#      ("Ich bin stolz auf das, was ich geschafft habe.", 0),
#      ("Das Konzert gestern war mega! Ich freuemich aufs nächstes.", )
#  ]
# test_set = [
#      ("Ich freue mich auf das Wochenende!", 0),
#      ("Das war ein großartiger Film.", 0),
#      ("Endlich Urlaub, ich kann es kaum erwarten!", 0),
#      ("Danke für deine Hilfe, das bedeutet mir viel.", 0),
#     ("Heute war ein wunderschöner Tag im Park.", 0),
#      ("Ich bereue alles, was ich getan habe.", -1),
#      ("Alles läuft schief, wie immer, enttäuscht.", -1),
#      ("Ich bin total frustriert. Ich hasse es.", -1),
#      ("Mir geht es schlecht.", -1),
#      ("Leute sind so unfreundlich.", -1)
#  ]
    # Vorverarbeitung
    #Frage: Muss ich den Pfad so angeben, auch wenn ich es oben importiert habe??? Das kommt mir suspekt vor...
train = [{"tokens": tokenizer.simple_tokenize(text), "label": int(label)} for text, label in train_set]
test = [{"tokens": tokenizer.simple_tokenize(text), "label": int(label)} for text, label in test_set]


    # TF-IDF Training
model = preprocessing.tfidfModel.TfIdfModel()
model.train([entry["tokens"] for entry in train])

# Feature-Vektoren
X_train = np.array([model.doc_vector(entry["tokens"]) for entry in train])
y_train = np.array([entry["label"] for entry in train])
X_test = np.array([model.doc_vector(entry["tokens"]) for entry in test])
y_test = np.array([entry["label"] for entry in test])

#lineares SVM ausführen
runLinearSVM.runLinearSVM(X_train,X_test,y_train,y_test)

#MLP ausführen
runMLP.runMPL(X_train,X_test,y_train,y_test)



