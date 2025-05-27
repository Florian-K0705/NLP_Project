from data.data_utils import GoEmotionsDataset

import ast

data_path = "C:/Users/MEIN PC/Downloads/data/goEmotions"

dataset = GoEmotionsDataset(path=data_path, split="train")

for i in range(10):
    text, label = dataset[i]
    print(f"Beispiel {i+1}:")
    print(f"Text: {text}")
    print(f"Label(s): {label}")
    print("-" * 40)

X_train = dataset.corpus
y_train = [ast.literal_eval(label) for label in dataset.labels]