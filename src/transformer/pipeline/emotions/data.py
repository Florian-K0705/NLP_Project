import pandas as pd
from datasets import Dataset, interleave_datasets
from transformers import AutoTokenizer
import os
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import numpy as np
from scipy.special import softmax


if os.path.exists("./fine_tuned_goemotions"):
    neuroBert_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_emotions")
else:
    neuroBert_tokenizer = AutoTokenizer.from_pretrained("boltuix/NeuroBERT-Small")



def tokenize_function(examples):
    return neuroBert_tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)


def load_traindata(tokenized=True):
    train_data = pd.read_csv("/home/florian/Dokumente/Programme/H-BRS/1. Semester/Natural Language Processing/NLP_Project/data/Emotions/train.tsv", sep="\t")

    y = train_data["label"].to_numpy()
    class_weights = softmax(compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y))


    dataset1 = Dataset.from_pandas(train_data.query("label == 0"))
    dataset2 = Dataset.from_pandas(train_data.query("label == 1"))
    dataset3 = Dataset.from_pandas(train_data.query("label == 2"))
    dataset4 = Dataset.from_pandas(train_data.query("label == 3"))
    dataset5 = Dataset.from_pandas(train_data.query("label == 4"))
    dataset6 = Dataset.from_pandas(train_data.query("label == 5"))


    data = interleave_datasets([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6], stopping_strategy="all_exhausted")


    #data = Dataset.from_pandas(train_data)

    tokenized_dataset = data.map(tokenize_function, batched=True, num_proc=8)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    if tokenized:
        return tokenized_dataset
    else:
        return data
    

def load_valdata(tokenized=True):
    validation_data = pd.read_csv("/home/florian/Dokumente/Programme/H-BRS/1. Semester/Natural Language Processing/NLP_Project/data/Emotions/val.tsv", sep="\t")

    data = Dataset.from_pandas(validation_data)

    tokenized_dataset = data.map(tokenize_function, batched=True, num_proc=8)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    if tokenized:
        return tokenized_dataset
    else:
        return data
    

def load_testdata(tokenized=True):
    test_data = pd.read_csv("/home/florian/Dokumente/Programme/H-BRS/1. Semester/Natural Language Processing/NLP_Project/data/Emotions/test.tsv", sep="\t")

    data = Dataset.from_pandas(test_data)

    tokenized_dataset = data.map(tokenize_function, batched=True,  num_proc=8)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])


    if tokenized:
        return tokenized_dataset
    else:
        return data
    

if __name__ == "__main__":
    
    data = load_traindata()

    print(data)