from transformers import Trainer, TrainingArguments, BertForSequenceClassification
from data import *
import os


def train():

    batch_size = 256
    learning_rate = 2e-5
    epochs = 150



    ########################


    if os.path.exists("./fine_tuned_goemotions"):
        model = BertForSequenceClassification.from_pretrained("./fine_tuned_goemotions", num_labels=28)
    else:
        model = BertForSequenceClassification.from_pretrained("boltuix/NeuroBERT-Small", num_labels=28)

    train_dataset = load_traindata()
    val_dataset = load_valdata()

    training_args = TrainingArguments (
        output_dir="./goEmotions_results",
        num_train_epochs=epochs,  # Increased epochs for small dataset
        per_device_train_batch_size=batch_size,
        logging_dir="./goEmotions_logs",
        logging_steps=100,
        logging_strategy="steps",
        save_steps=100,
        learning_rate=learning_rate,  # Adjusted for NeuroBERT-Small
        eval_strategy="steps",
        per_device_eval_batch_size=batch_size
    )

    trainer = Trainer (
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()


    model.save_pretrained("./fine_tuned_neurobert")
    neuroBert_tokenizer.save_pretrained("./fine_tuned_neurobert")


if __name__ == "__main__":
    train()
    print("Training complete. Model and tokenizer saved.")