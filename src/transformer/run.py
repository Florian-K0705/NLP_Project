import torch
from nltk.tokenize import word_tokenize

import data.data_utils as data_utils
from transformer.models import BasicTransformerModel, BertModel, NeuroBertSmallModel
from transformer.train import train,train2
from transformer.evaluate import test

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocessing.embedding import Embedding

    



def main():

      path_goemotions = "/home/florian/Dokumente/Programme/H-BRS/1. Semester/Natural Language Processing/NLP_Project/data/goEmotions/data"
      path_emotions = "/home/florian/Dokumente/Programme/H-BRS/1. Semester/Natural Language Processing/NLP_Project/data/Emotions"

      train_ds = data_utils.GoEmotionsDataset(path=path_goemotions, split="train")
      val_ds = data_utils.GoEmotionsDataset(path=path_goemotions, split="val")

      #train_ds = data_utils.EmotionsDataset(path=path_emotions, split="train")
      #val_ds = data_utils.EmotionsDataset(path=path_emotions, split="val")

      num_classes = 28

      ############################################

      #model = BasicTransformerModel(num_layers=5, num_heads=3, feature_dim=300, num_classes=num_classes)
      #model = BertModel(num_classes=num_classes)

      model = NeuroBertSmallModel(num_classes=num_classes)


      ############################################

      # Start training    
      train2(model, train_ds,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.000001), 
            criterion=torch.nn.CrossEntropyLoss(), 
            #batch_size=4096,
            device="cuda", 
            num_classes=num_classes,
            num_epochs=25)
      
      # Start testing
      print("Start testing\n")

      print("Test on Trainingset")
      test(model, train_ds, device="cuda")

      print("Test on Validationset")
      test(model, val_ds, device="cuda")

      torch.save(model.state_dict(), "/home/florian/Dokumente/Programme/H-BRS/1. Semester/Natural Language Processing/NLP_Project/data/model.pth")