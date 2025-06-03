from transformers import AutoTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("./fine_tuned_goemotions", num_labels=28)
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_goemotions")

classes = ["admiration",
"amusement",
"anger",
"annoyance",
"approval",
"caring",
"confusion",
"curiosity",
"desire",
"disappointment",
"disapproval",
"disgust",
"embarrassment",
"excitement",
"fear",
"gratitude",
"grief",
"joy",
"love",
"nervousness",
"optimism",
"pride",
"realization",
"relief",
"remorse",
"sadness",
"surprise",
"neutral"]

while (True):

    input_string = str(input("Gib einen Satz ein: "))

    if input_string == "quit":
        break

    tokens = tokenizer(input_string, return_tensors="pt")

    probs = torch.nn.functional.softmax(model(**tokens).logits, dim=1)

    prediction = torch.argmax(probs).item()
    print(classes[prediction])