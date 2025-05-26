import torch
from nltk.tokenize import word_tokenize

import data.data_utils as data_utils
from transformer.models import BasicTransformerModel, BertModel
from transformer.train import train
from transformer.evaluate import test

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocessing.embedding import Embedding

    



def main():

      path = "/home/florian/Dokumente/Programme/H-BRS/1. Semester/Natural Language Processing/NLP_Project/data/goEmotions/data"

      model = BasicTransformerModel(num_layers=3, num_heads=2, feature_dim=50, num_classes=28)

      #model = BertModel(num_classes=28)


      train_ds = data_utils.GoEmotionsDataset(path=path, split="train")
      val_ds = data_utils.GoEmotionsDataset(path=path, split="val")


      # Start training    
      train(model, train_ds,
            optimizer=torch.optim.Adam(model.bert.classifier.parameters(), lr=0.000001), 
            criterion=torch.nn.CrossEntropyLoss(), 
            batch_size=4096,
            device="cuda", 
            num_classes=28,
            num_epochs=10)
      
      # Start testing
      print("Start testing\n")

      print("Test on Trainingset")
      test(model, train_ds, device="cuda")

      print("Test on Validationset")
      test(model, val_ds, device="cuda")