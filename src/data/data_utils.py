import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset



## Die abstrakte Klasse Data definiert alle Methoden, die wir zum Laden der Daten brauchen. Dabei sollen die Daten in 
## beliegiger Form (z.B. numpy array, torch Dataset) zurückgegeben werden können.
class Data():
    def __init__(self, path):
        pass

    def get_numpy_data():
        pass

    def get_pytorch_dataset():
        pass

## Die abstrakte Klasse MyDataset definiert wie die Pytorch Dataset Klasse aufgebaut ist.
## Sie wird von den konkreten Dataset-Klassen (z.B. GoEmotionsDataset) geerbt.
## Letztlich brauchen wir nur die MyDataset-Klasse, um die Daten in einem Pytorch Dataset-Format zurückzugeben, aber der übersichtilicheit halber
## haben wir die MyDataset-Klasse und die GoEmotionsDataset-Klasse, etc. getrennt.
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        return text, label



##############################################################################################################################################
###########################################         GoEmotions         #######################################################################
##############################################################################################################################################




class GoEmotionsDataset(MyDataset):
    def __init__(self, data, labels):
        super().__init__(data, labels)


class GoEmotions(Data):

    ## TODO
    # Es gibt Sätze wie: "I'm sure she was just jealous of [NAME]...", wo eine Maske wie [NAME] vorkommt.
    # Das ist ein Problem, weil wir die Daten nicht so einfach tokenisieren können.
    # denn aus [NAME] wird dann { '[', 'NAME', ']', '...' }
    

    def __init__(self, path, dataset_type="train"):

        self.dataset_type = dataset_type
        
        train_path = os.path.join(path, "train.tsv")
        test_path = os.path.join(path, "test.tsv")
        val_path = os.path.join(path, "dev.tsv")

        self.train_data = pd.read_csv(train_path, sep="\t")
        self.test_data = pd.read_csv(test_path, sep="\t")
        self.val_data = pd.read_csv(val_path, sep="\t")

        self.classes = open(os.path.join(path, "emotions.txt"), "r").read().splitlines()

    def get_numpy_data(self):

        if self.dataset_type == "train":
            corpus = self.train_data["text"].tolist()
            labels = self.train_data["labels"].tolist()
        elif self.dataset_type == "test":
            corpus = self.test_data["text"].tolist()
            labels = self.test_data["labels"].tolist()
        elif self.dataset_type == "val":
            corpus = self.val_data["text"].tolist()
            labels = self.val_data["labels"].tolist()
        else:
            raise ValueError("dataset_type must be one of ['train', 'test', 'val']")
        
        return corpus, labels


    def get_pytorch_dataset(self):
        
        if self.dataset_type == "train":
            corpus = self.train_data["text"].tolist()
            labels = self.train_data["labels"].tolist()
        elif self.dataset_type == "test":
            corpus = self.test_data["text"].tolist()
            labels = self.test_data["labels"].tolist()
        elif self.dataset_type == "val":
            corpus = self.val_data["text"].tolist()
            labels = self.val_data["labels"].tolist()
        else:
            raise ValueError("dataset_type must be one of ['train', 'test', 'val']")
        
        return GoEmotionsDataset(corpus, labels)







##############################################################################################################################################
###############################################         Emotions       #######################################################################
##############################################################################################################################################


## TODO
class Emotions(Data):

    def __init__(self, path):
        pass

    def get_numpy_data():
        pass

    def get_pytorch_dataset():
        pass








##############################################################################################################################################
###############################################         App Reviews         ##################################################################
##############################################################################################################################################


## TODO
class AppReviews(Data):

    def __init__(self, path):
        pass

    def get_numpy_data():
        pass

    def get_pytorch_dataset():
        pass

if __name__ == "__main__":

    data = GoEmotions(path="/home/florian/Dokumente/Programme/H-BRS/1. Semester/Natural Language Processing/NLP_Project/data/goEmotions/data")
    x, y = data.get_numpy_data()
    dataset = data.get_pytorch_dataset()

    hist = np.zeros(len(data.classes))

    for i in range(len(x)):
        hist[dataset[i][1]] += 1


    print(hist)
    print(np.sum(hist))
    print(len(dataset))
