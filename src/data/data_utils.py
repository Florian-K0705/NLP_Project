import os
import pandas as pd
from torch.utils.data import Dataset



##############################################################################################################################################
###########################################         GoEmotions         #######################################################################
##############################################################################################################################################



class GoEmotionsDataset(Dataset):

    def __init__(self, path, split="train"):
        super().__init__()


        self.classes = open(os.path.join(path, "emotions.txt"), "r").read().splitlines()

        if split == "train":

            train_path = os.path.join(path, "train.tsv")

            train_data = pd.read_csv(train_path, sep="\t")

            self.corpus = train_data["text"].tolist()
            self.labels = train_data["labels"].tolist()

            self.length = len(self.corpus)

        elif split == "test":

            test_path = os.path.join(path, "test.tsv")
            
            test_data = pd.read_csv(test_path, sep="\t")

            self.corpus = test_data["text"].tolist()
            self.labels = test_data["labels"].tolist()

            self.length = len(self.corpus)

        elif split == "val":

            val_path = os.path.join(path, "val.tsv")
            
            val_data = pd.read_csv(val_path, sep="\t")

            self.corpus = val_data["text"].tolist()
            self.labels = val_data["labels"].tolist()

            self.length = len(self.corpus)
        else:
            raise ValueError("dataset_type must be one of ['train', 'test', 'val']")
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        text = self.corpus[idx]
        label = self.labels[idx]

        return text, label





##############################################################################################################################################
###############################################         Emotions       #######################################################################
##############################################################################################################################################

class EmotionsDataset(Dataset):

    def __init__(self, path, split="train"):
        super().__init__()

        self.classes = open(os.path.join(path, "emotions.txt"), "r").read().splitlines()

        if split == "train":

            train_path = os.path.join(path, "train.tsv")

            train_data = pd.read_csv(train_path, sep="\t")

            self.corpus = train_data["text"].tolist()
            self.labels = train_data["label"].tolist()

            self.length = len(self.corpus)

        elif split == "test":

            test_path = os.path.join(path, "test.tsv")
            
            test_data = pd.read_csv(test_path, sep="\t")

            self.corpus = test_data["text"].tolist()
            self.labels = test_data["label"].tolist()

            self.length = len(self.corpus)

        elif split == "val":

            val_path = os.path.join(path, "val.tsv")
            
            val_data = pd.read_csv(val_path, sep="\t")

            self.corpus = val_data["text"].tolist()
            self.labels = val_data["label"].tolist()

            self.length = len(self.corpus)
        else:
            raise ValueError("dataset_type must be one of ['train', 'test', 'val']")
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        text = self.corpus[idx]
        label = self.labels[idx]

        return text, label


##############################################################################################################################################
###############################################         App Reviews         ##################################################################
##############################################################################################################################################


class AppReviewsDataset(Dataset):

    def __init__(self, path, split="train"):
        super().__init__()

        if split == "train":

            train_path = os.path.join(path, "train.tsv")

            train_data = pd.read_csv(train_path, sep="\t")

            self.corpus = train_data["review"].tolist()
            self.labels = train_data["star"].tolist()

            self.length = len(self.corpus)

        elif split == "test":

            test_path = os.path.join(path, "test.tsv")
            
            test_data = pd.read_csv(test_path, sep="\t")

            self.corpus = test_data["review"].tolist()
            self.labels = test_data["star"].tolist()

            self.length = len(self.corpus)

        elif split == "val":

            val_path = os.path.join(path, "val.tsv")
            
            val_data = pd.read_csv(val_path, sep="\t")

            self.corpus = val_data["review"].tolist()
            self.labels = val_data["star"].tolist()

            self.length = len(self.corpus)
        else:
            raise ValueError("dataset_type must be one of ['train', 'test', 'val']")
        

    def __len__(self):
        return self.length
    

    def __getitem__(self, idx):
        text = self.corpus[idx]
        label = self.labels[idx]

        return text, label
