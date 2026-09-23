import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


FEEDBACK_FILE = "data/feedback.csv"


# -----------------------------
# Load feedback
# -----------------------------
df = pd.read_csv(FEEDBACK_FILE)

prediction = df["prediction"]
actual = df["actual"]


# -----------------------------
# Calculate metrics
# -----------------------------
accuracy = accuracy_score(actual, prediction)
precision = precision_score(actual, prediction, zero_division=0)
recall = recall_score(actual, prediction, zero_division=0)
f1 = f1_score(actual, prediction, zero_division=0)

total = len(df)
errors = (prediction != actual).sum()


# -----------------------------
# Display monitoring results
# -----------------------------
print("--------------------------------")
print("      MODEL MONITORING")
print("--------------------------------")

print("Model Version :", df["model_version"].iloc[-1])
print("Total Predictions :", total)
print("Correct Predictions:", total - errors)
print("Incorrect Predictions:", errors)

print()
print("Performance")
print("----------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# -----------------------------
# Self-adaptation threshold
# -----------------------------
RECALL_THRESHOLD = 0.70

print()
print("Adaptation Check")
print("----------------")

if recall < RECALL_THRESHOLD:
    print("⚠ Performance degradation detected.")
    print("→ Retraining should be triggered.")
else:
    print("✓ Model performance is healthy.")
    print("→ No retraining required.")