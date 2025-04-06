import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Read the file
filepath= r"C:\Users\amish\Documents\open\DropoutUncertaintyExps\UCI_Datasets\bostonHousing\results\validation_ll_5_xepochs_2_hidden_layers.txt"
with open( filepath, "r") as file:
    lines = file.readlines()

# Parse the data
data = []
for line in lines:
    match = re.search(r"Dropout_Rate: ([0-9.]+) Tau: ([0-9.]+) :: (-?[0-9.]+)", line)
    if match:
        dropout_rate = float(match.group(1))
        tau = float(match.group(2))
        value = float(match.group(3))
        data.append((dropout_rate, tau, value))

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Dropout Rate", "Tau", "Value"])

# Pivot to get heatmap format
heatmap_data = df.pivot(index="Dropout Rate", columns="Tau", values="Value")

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5)
plt.title("Heatmap of Dropout Rate vs Tau")
plt.xlabel("Tau")
plt.ylabel("Dropout Rate")
plt.show()
