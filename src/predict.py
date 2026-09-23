import joblib
import pandas as pd

ACTIVE_MODEL_FILE = "models/active_model.txt"


def load_active_model():

    with open(ACTIVE_MODEL_FILE, "r") as f:
        model_version = f.read().strip()

    model_path = f"models/model_{model_version.lower()}.pkl"

    model = joblib.load(model_path)

    return model, model_version


def predict_machine(data):

    model, model_version = load_active_model()

    df = pd.DataFrame([data])

    probability = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]

    return model_version, prediction, probability


data = {
    "Type": "M",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 0
}


model_version, prediction, probability = predict_machine(data)

print("--------------------------------")
print("     MACHINE PREDICTION")
print("--------------------------------")

print("Active Model :", model_version)

if prediction == 1:
    print("Machine Status : FAILURE")
else:
    print("Machine Status : NORMAL")

print(f"Failure Probability : {probability * 100:.2f}%")