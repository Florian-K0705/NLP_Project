import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import os

if os.path.exists("./fine_tuned_goemotions"):
    neuroBert_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_goemotions")
else:
    neuroBert_tokenizer = AutoTokenizer.from_pretrained("boltuix/NeuroBERT-Small")


def tokenize_function(examples):
    return neuroBert_tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)


def load_traindata(tokenized=True):
    train_data = pd.read_csv("/home/florian/Dokumente/Programme/H-BRS/1. Semester/Natural Language Processing/NLP_Project/data/goEmotions/data/train.tsv", sep="\t")

    data = Dataset.from_pandas(train_data)

    tokenized_dataset = data.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    tokenized_dataset = tokenized_dataset.rename_column("labels", "label")

    if tokenized:
        return tokenized_dataset
    else:
        return data
    

def load_valdata(tokenized=True):
    validation_data = pd.read_csv("/home/florian/Dokumente/Programme/H-BRS/1. Semester/Natural Language Processing/NLP_Project/data/goEmotions/data/val.tsv", sep="\t")

    data = Dataset.from_pandas(validation_data)

    tokenized_dataset = data.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    tokenized_dataset = tokenized_dataset.rename_column("labels", "label")

    if tokenized:
        return tokenized_dataset
    else:
        return data


def load_testdata(tokenized=True):
    test_data = pd.read_csv("/home/florian/Dokumente/Programme/H-BRS/1. Semester/Natural Language Processing/NLP_Project/data/goEmotions/data/test.tsv", sep="\t")

    data = Dataset.from_pandas(test_data)

    tokenized_dataset = data.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    tokenized_dataset = tokenized_dataset.rename_column("labels", "label")

    if tokenized:
        return tokenized_dataset
    else:
        return data


if __name__ == "__main__":

    train_ds = load_traindata()