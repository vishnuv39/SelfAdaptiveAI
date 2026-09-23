import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split


DATA_FILE = "data/ai4i2020.csv"
MODEL_FILE = "models/model_v1.pkl"
FEEDBACK_FILE = "data/feedback.csv"


# -----------------------------
# Load dataset
# -----------------------------
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


# -----------------------------
# Use the same test split
# as train.py
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Load Model V1
# -----------------------------
model = joblib.load(MODEL_FILE)


# -----------------------------
# Generate predictions
# -----------------------------
predictions = model.predict(X_test)


# -----------------------------
# Create feedback data
# -----------------------------
feedback = X_test.copy()

feedback["prediction"] = predictions
feedback["actual"] = y_test.values
feedback["model_version"] = "V1"
feedback["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------
# Save feedback
# -----------------------------
if os.path.exists(FEEDBACK_FILE):

    feedback.to_csv(
        FEEDBACK_FILE,
        mode="a",
        header=False,
        index=False
    )

else:

    feedback.to_csv(
        FEEDBACK_FILE,
        index=False
    )


print("Real feedback generated successfully.")
print("Number of feedback records:", len(feedback))
print("Correct predictions:",
      (feedback["prediction"] == feedback["actual"]).sum())
print("Incorrect predictions:",
      (feedback["prediction"] != feedback["actual"]).sum())
print("Saved to:", FEEDBACK_FILE)