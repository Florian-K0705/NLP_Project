from transformers import AutoTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("./fine_tuned_emotions", num_labels=6)
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_emotions")

classes = ["sadness", "joy" ,"love", "anger", "fear", "surprise"]

while (True):

    input_string = str(input("Gib einen Satz ein: "))

    if input_string == "quit":
        break

    tokens = tokenizer(input_string, return_tensors="pt")

    probs = torch.nn.functional.softmax(model(**tokens).logits, dim=1)

    prediction = torch.argmax(probs).item()
    print(classes[prediction])
