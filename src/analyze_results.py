import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv("results.csv")
print("=== Inference Results ===")
print(df.head())

y_true = df["true_label"]
y_pred = df["pred_label"]

acc = accuracy_score(y_true, y_pred)
print(f"\nTotal Accuracy: {acc*100:.2f}%")

cm = confusion_matrix(y_true, y_pred)
labels = ["Setosa", "Versicolor", "Virginica"]

print("\n=== Confusion Matrix ===")
print(pd.DataFrame(cm, index=labels, columns=labels))

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=labels))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (Accuracy = {acc*100:.2f}%)")
plt.tight_layout()
plt.savefig("results_visualization.png", dpi=300)
