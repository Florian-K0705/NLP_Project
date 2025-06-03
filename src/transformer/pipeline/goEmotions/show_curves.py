import matplotlib.pyplot as plt
import pandas as pd

path = "./Run1"

train_loss = pd.read_csv(f"{path}/train_loss_curve.csv")
val_loss = pd.read_csv(f"{path}/val_loss_curve.csv")


train_values = train_loss["Value"].tolist()
val_values = val_loss["Value"].tolist()

plt.plot(train_loss["Step"], train_values, label="Train Loss")
plt.plot(val_loss["Step"], val_values, label="Validation Loss")

plt.show()