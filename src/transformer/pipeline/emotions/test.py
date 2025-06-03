from data import load_valdata, load_testdata
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, top_k_accuracy_score, ConfusionMatrixDisplay, confusion_matrix, balanced_accuracy_score, recall_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

test_dataset = load_valdata(tokenized=False)
classes = ["sadness", "joy" ,"love", "anger", "fear", "surprise"]

model = BertForSequenceClassification.from_pretrained("./fine_tuned_emotions", num_labels=6)
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_emotions")

real_labels = []
pred_labels = []
probabilities = []

print("Evaluating model on validation set...")
for i in tqdm(range(len(test_dataset))):

    label = test_dataset[i]["label"]
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
print(f"Balanced Accuracy on validation set: {balanced_accuracy_score(real_labels, pred_labels) * 100:.2f}%")

recall_values = recall_score(real_labels, pred_labels, average=None)

for i, c in enumerate(classes):
    print(f"Recall of {c} is {recall_values[i]}")

cm = confusion_matrix(real_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot()
plt.show()
#print(f"Top-{k0}-Accuracy on validation set: {top_k_accuracy_score(real_labels, probabilities, k=k0) * 100:.2f}%")
#print(f"Top-{k1}-Accuracy on validation set: {top_k_accuracy_score(real_labels, probabilities, k=k1) * 100:.2f}%")
#print(f"Top-{k2}-Accuracy on validation set: {top_k_accuracy_score(real_labels, probabilities, k=k2) * 100:.2f}%")