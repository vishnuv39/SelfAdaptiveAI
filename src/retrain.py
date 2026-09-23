import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


DATA_FILE = "data/ai4i2020.csv"
FEEDBACK_FILE = "data/feedback.csv"
MODEL_FILE = "models/model_v2.pkl"


# --------------------------------
# 1. Load original dataset
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
# 2. Load feedback
# --------------------------------

feedback = pd.read_csv(FEEDBACK_FILE)

feedback_X = feedback[features]
feedback_y = feedback["actual"]


print("--------------------------------")
print("     FEEDBACK INFORMATION")
print("--------------------------------")

print("Feedback records:", len(feedback))


# --------------------------------
# 3. Combine original + feedback
# --------------------------------

X_combined = pd.concat(
    [X, feedback_X],
    ignore_index=True
)

y_combined = pd.concat(
    [y, feedback_y],
    ignore_index=True
)


print("Original records :", len(X))
print("Combined records :", len(X_combined))


# --------------------------------
# 4. Split updated dataset
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_combined,
    y_combined,
    test_size=0.2,
    random_state=42,
    stratify=y_combined
)


# --------------------------------
# 5. Preprocessing
# --------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            ["Type"]
        )
    ],
    remainder="passthrough"
)


# --------------------------------
# 6. Model V2
# --------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------
# 7. MLflow
# --------------------------------

mlflow.set_experiment("Predictive_Maintenance")


with mlflow.start_run() as run:

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    print("\n--------------------------------")
    print("       MODEL V2 TRAINING")
    print("--------------------------------")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


    # --------------------------------
    # MLflow parameters
    # --------------------------------

    mlflow.log_param(
        "model_version",
        "V2"
    )

    mlflow.log_param(
        "model",
        "RandomForest"
    )

    mlflow.log_param(
        "n_estimators",
        300
    )

    mlflow.log_param(
        "max_depth",
        12
    )

    mlflow.log_param(
        "class_weight",
        "balanced"
    )

    mlflow.log_param(
        "training_records",
        len(X_combined)
    )

    mlflow.log_param(
        "feedback_records",
        len(feedback)
    )


    # --------------------------------
    # MLflow metrics
    # --------------------------------

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )


    # --------------------------------
    # Log model
    # --------------------------------

    mlflow.sklearn.log_model(
        pipeline,
        name="predictive_maintenance_model_v2",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )


    print("\nMLflow Run ID:", run.info.run_id)


# --------------------------------
# 8. Save Model V2
# --------------------------------

joblib.dump(
    pipeline,
    MODEL_FILE
)

print("\nModel V2 saved to:", MODEL_FILE)