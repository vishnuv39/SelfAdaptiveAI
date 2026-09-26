import os
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
ACTIVE_MODEL_FILE = "models/active_model.txt"

RECALL_THRESHOLD = 0.70


# --------------------------------
# Get active model
# --------------------------------

def get_active_model():

    with open(ACTIVE_MODEL_FILE, "r") as file:
        return file.read().strip()


# --------------------------------
# Get next model version
# --------------------------------

def get_next_model_version():

    model_files = [
        file for file in os.listdir("models")
        if file.startswith("model_v")
        and file.endswith(".pkl")
    ]

    versions = []

    for file in model_files:

        try:
            version = int(
                file.replace("model_v", "").replace(".pkl", "")
            )

            versions.append(version)

        except ValueError:
            pass

    if not versions:
        return 1

    return max(versions) + 1


# --------------------------------
# 1. Load feedback
# --------------------------------

df = pd.read_csv(FEEDBACK_FILE)

prediction = df["prediction"]
actual = df["actual"]

current_model = get_active_model()


# --------------------------------
# 2. Monitor current performance
# --------------------------------

accuracy = accuracy_score(
    actual,
    prediction
)

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
    # 4. Determine next model version
    # =================================

    next_version = get_next_model_version()

    candidate_model = f"V{next_version}"
    candidate_file = f"models/model_v{next_version}.pkl"

    print(f"\nCandidate Model : {candidate_model}")

    # =================================
    # 5. Run retraining automatically
    # =================================

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    env = os.environ.copy()

    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        [sys.executable, "src/retrain.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )

    print("\nRetraining Output")
    print("------------------")
    print(result.stdout)

    if result.returncode != 0:

        print("❌ Retraining failed.")
        print(result.stderr)

        selected_model = current_model

    elif not os.path.exists(candidate_file):

        print(
            f"❌ Expected candidate model was not created: "
            f"{candidate_file}"
        )

        selected_model = current_model

    else:

        print("✓ Retraining completed.")

        # =================================
        # 6. Load dataset for evaluation
        # =================================

        data = pd.read_csv(
            "data/ai4i2020.csv"
        )

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
        # 7. Load active and candidate models
        # =================================

        active_file = (
            f"models/model_"
            f"{current_model.lower()}.pkl"
        )

        candidate_file = (
            f"models/model_"
            f"{candidate_model.lower()}.pkl"
        )

        model_active = joblib.load(
            active_file
        )

        model_candidate = joblib.load(
            candidate_file
        )

        pred_active = model_active.predict(
            X_test
        )

        pred_candidate = model_candidate.predict(
            X_test
        )

        # =================================
        # 8. Evaluate active model
        # =================================

        active_recall = recall_score(
            y_test,
            pred_active,
            zero_division=0
        )

        active_f1 = f1_score(
            y_test,
            pred_active,
            zero_division=0
        )

        # =================================
        # 9. Evaluate candidate model
        # =================================

        candidate_recall = recall_score(
            y_test,
            pred_candidate,
            zero_division=0
        )

        candidate_f1 = f1_score(
            y_test,
            pred_candidate,
            zero_division=0
        )

        print("\nModel Comparison")
        print("----------------")

        print(
            f"{current_model} Recall : "
            f"{active_recall:.4f}"
        )

        print(
            f"{current_model} F1     : "
            f"{active_f1:.4f}"
        )

        print(
            f"{candidate_model} Recall : "
            f"{candidate_recall:.4f}"
        )

        print(
            f"{candidate_model} F1     : "
            f"{candidate_f1:.4f}"
        )

        # =================================
        # 10. Select model
        # =================================

        if (
            candidate_recall >= RECALL_THRESHOLD
            and candidate_f1 >= active_f1
        ):

            selected_model = candidate_model

            print(
                f"\n✓ {candidate_model} "
                f"passed deployment criteria."
            )

            print(
                f"→ {candidate_model} activated."
            )

        else:

            selected_model = current_model

            print(
                f"\n✗ {candidate_model} "
                f"failed deployment criteria."
            )

            print(
                f"→ {current_model} remains active."
            )


# =================================
# 11. Save active model
# =================================

with open(
    ACTIVE_MODEL_FILE,
    "w"
) as file:

    file.write(selected_model)


print("\n================================")
print("Active Model:", selected_model)
print("================================")