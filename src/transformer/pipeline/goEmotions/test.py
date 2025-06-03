from data import load_valdata, load_testdata
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import numpy as np
from tqdm import tqdm

test_dataset = load_valdata(tokenized=False)

#model = BertForSequenceClassification.from_pretrained("./fine_tuned_goemotions", num_labels=28)
#tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_goemotions")

model = BertForSequenceClassification.from_pretrained("./goEmotions_results/checkpoint-3100", num_labels=28)
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_goemotions")

real_labels = []
pred_labels = []
probabilities = []

print("Evaluating model on validation set...")
for i in tqdm(range(len(test_dataset))):

    label = test_dataset[i]["labels"]
    text = test_dataset[i]["text"]

    tokens = tokenizer(text, return_tensors="pt")

    out = model(**tokens)

    real_labels.append(label)
    pred_labels.append(out.logits.argmax(dim=1).item())
    probabilities.append(out.logits.softmax(dim=1).detach().numpy()[0])


k0 = 2
k1 = 3
k2 = 5


print(f"Accuracy on validation set: {accuracy_score(real_labels, pred_labels) * 100:.2f}%")
print(f"Top-{k0}-Accuracy on validation set: {top_k_accuracy_score(real_labels, probabilities, k=k0) * 100:.2f}%")
print(f"Top-{k1}-Accuracy on validation set: {top_k_accuracy_score(real_labels, probabilities, k=k1) * 100:.2f}%")
print(f"Top-{k2}-Accuracy on validation set: {top_k_accuracy_score(real_labels, probabilities, k=k2) * 100:.2f}%")