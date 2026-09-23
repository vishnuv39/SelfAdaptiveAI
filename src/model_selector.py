import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pandas as pd


DATA_FILE = "data/ai4i2020.csv"

MODEL_V1 = "models/model_v1.pkl"
MODEL_V2 = "models/model_v2.pkl"


# --------------------------------
# Load dataset
# --------------------------------
df = pd.read_csv(DATA_FILE)

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]


# --------------------------------
# Use the same test data
# --------------------------------
from sklearn.model_selection import train_test_split

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# Load both models
# --------------------------------
model_v1 = joblib.load(MODEL_V1)
model_v2 = joblib.load(MODEL_V2)


# --------------------------------
# Evaluate V1
# --------------------------------
pred_v1 = model_v1.predict(X_test)

v1_recall = recall_score(
    y_test,
    pred_v1,
    zero_division=0
)

v1_f1 = f1_score(
    y_test,
    pred_v1,
    zero_division=0
)

v1_accuracy = accuracy_score(
    y_test,
    pred_v1
)


# --------------------------------
# Evaluate V2
# --------------------------------
pred_v2 = model_v2.predict(X_test)

v2_recall = recall_score(
    y_test,
    pred_v2,
    zero_division=0
)

v2_f1 = f1_score(
    y_test,
    pred_v2,
    zero_division=0
)

v2_accuracy = accuracy_score(
    y_test,
    pred_v2
)


# --------------------------------
# Display comparison
# --------------------------------
print("--------------------------------")
print("       MODEL COMPARISON")
print("--------------------------------")

print("\nModel V1")
print(f"Accuracy : {v1_accuracy:.4f}")
print(f"Recall   : {v1_recall:.4f}")
print(f"F1 Score : {v1_f1:.4f}")

print("\nModel V2")
print(f"Accuracy : {v2_accuracy:.4f}")
print(f"Recall   : {v2_recall:.4f}")
print(f"F1 Score : {v2_f1:.4f}")


# --------------------------------
# Deployment criteria
# --------------------------------
RECALL_THRESHOLD = 0.70

print("\nDeployment Check")
print("----------------")


if (
    v2_recall >= RECALL_THRESHOLD
    and v2_f1 >= v1_f1
):
    selected_model = "V2"
    selected_path = MODEL_V2

else:
    selected_model = "V1"
    selected_path = MODEL_V1


print("Selected Model:", selected_model)


# --------------------------------
# Save selected model information
# --------------------------------
with open("models/active_model.txt", "w") as file:
    file.write(selected_model)


print("Active model saved.")