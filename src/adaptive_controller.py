import pandas as pd
import joblib
import subprocess
import sys

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split


FEEDBACK_FILE = "data/feedback.csv"

MODEL_V1 = "models/model_v1.pkl"
MODEL_V2 = "models/model_v2.pkl"

ACTIVE_MODEL_FILE = "models/active_model.txt"

RECALL_THRESHOLD = 0.70


# --------------------------------
# 1. Load feedback
# --------------------------------

df = pd.read_csv(FEEDBACK_FILE)

prediction = df["prediction"]
actual = df["actual"]

current_model = df["model_version"].iloc[-1]


# --------------------------------
# 2. Monitor current performance
# --------------------------------

accuracy = accuracy_score(actual, prediction)

precision = precision_score(
    actual,
    prediction,
    zero_division=0
)

recall = recall_score(
    actual,
    prediction,
    zero_division=0
)

f1 = f1_score(
    actual,
    prediction,
    zero_division=0
)


print("================================")
print("     SELF-ADAPTIVE CONTROLLER")
print("================================")

print("\nCurrent Model :", current_model)

print("\nCurrent Performance")
print("-------------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# =================================
# 3. Check degradation
# =================================

if recall >= RECALL_THRESHOLD:

    print("\n✓ Model performance is healthy.")
    print("→ No retraining required.")

    selected_model = current_model

else:

    print("\n⚠ Performance degradation detected.")
    print("→ Starting automatic retraining...")

    # =================================
    # 4. Run retraining automatically
    # =================================

    result = subprocess.run(
        [sys.executable, "src/retrain.py"],
        capture_output=True,
        text=True
    )

    print("\nRetraining Output")
    print("------------------")
    print(result.stdout)

    if result.returncode != 0:

        print("❌ Retraining failed.")
        print(result.stderr)

        selected_model = current_model

    else:

        print("✓ Retraining completed.")

        # =================================
        # 5. Load dataset for evaluation
        # =================================

        data = pd.read_csv("data/ai4i2020.csv")

        features = [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]

        X = data[features]
        y = data["Machine failure"]

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # =================================
        # 6. Load V1 and newly trained V2
        # =================================

        model_v1 = joblib.load(MODEL_V1)
        model_v2 = joblib.load(MODEL_V2)

        pred_v1 = model_v1.predict(X_test)
        pred_v2 = model_v2.predict(X_test)

        # =================================
        # 7. Evaluate V1
        # =================================

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

        # =================================
        # 8. Evaluate V2
        # =================================

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

        print("\nModel Comparison")
        print("----------------")
        print(f"V1 Recall : {v1_recall:.4f}")
        print(f"V1 F1     : {v1_f1:.4f}")
        print(f"V2 Recall : {v2_recall:.4f}")
        print(f"V2 F1     : {v2_f1:.4f}")

        # =================================
        # 9. Select model
        # =================================

        if (
            v2_recall >= RECALL_THRESHOLD
            and v2_f1 >= v1_f1
        ):

            selected_model = "V2"

            print("\n✓ V2 passed deployment criteria.")
            print("→ V2 activated.")

        else:

            selected_model = "V1"

            print("\n✗ V2 failed deployment criteria.")
            print("→ V1 remains active.")


# =================================
# 10. Save active model
# =================================

with open(ACTIVE_MODEL_FILE, "w") as file:
    file.write(selected_model)


print("\n================================")
print("Active Model:", selected_model)
print("================================")