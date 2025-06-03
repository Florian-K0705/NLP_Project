from transformers import Trainer, TrainingArguments, BertForSequenceClassification
from data import *
import os

def train():

    ########################################
    ########## Hyperparameter ##############
    ########################################
    

    batch_size = 256
    learning_rate = 2e-5
    epochs = 5



    ########################################
    ############# Training #################
    ########################################


    if os.path.exists("./fine_tuned_emotions"):
        model = BertForSequenceClassification.from_pretrained("./fine_tuned_emotions", num_labels=6)
    else:
        model = BertForSequenceClassification.from_pretrained("boltuix/NeuroBERT-Small", num_labels=6)


    train_dataset = load_traindata()
    val_dataset = load_valdata()


    training_args = TrainingArguments (
        output_dir="./emotions_results",
        num_train_epochs=epochs,  # Increased epochs for small dataset
        per_device_train_batch_size=batch_size,
        logging_dir="./emotions_logs",
        logging_steps=100,
        logging_strategy="steps",
        save_steps=100,
        learning_rate=learning_rate,  # Adjusted for NeuroBERT-Small
        eval_strategy="steps",
        per_device_eval_batch_size=batch_size,
        #  AUFPASSEN, dass das "beste" nicht das beste Ergebniss auf dem Trainingsdatensatz ist.
        load_best_model_at_end=True # Das beste Modell, das er während des Trainings findet, wird gespeichert. 
    )


    trainer = Trainer (
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Number of trainable Parameters:", trainer.get_num_trainable_parameters())

    trainer.train()

    model.save_pretrained("./fine_tuned_emotions")
    neuroBert_tokenizer.save_pretrained("./fine_tuned_emotions")

if __name__ == "__main__":
    train()
    print("Training complete. Model and tokenizer saved.")