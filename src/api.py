from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os
import joblib
import pandas as pd

app = FastAPI(
    title="Self-Adaptive Predictive Maintenance API",
    description="Predictive maintenance API using the active ML model.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ACTIVE_MODEL_FILE = "models/active_model.txt"


class MachineData(BaseModel):
    Type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float

class FeedbackData(BaseModel):
    Type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float

    prediction: int
    actual: int
    model_version: str

def load_active_model():

    with open(ACTIVE_MODEL_FILE, "r") as file:
        model_version = file.read().strip()

    model_path = f"models/model_{model_version.lower()}.pkl"

    model = joblib.load(model_path)

    return model, model_version


@app.get("/")
def home():

    return {
        "message": "Self-Adaptive Predictive Maintenance API",
        "status": "running"
    }


@app.get("/model")
def active_model():

    with open(ACTIVE_MODEL_FILE, "r") as file:
        model_version = file.read().strip()

    return {
        "active_model": model_version
    }


@app.post("/predict")
def predict(data: MachineData):

    model, model_version = load_active_model()

    input_data = pd.DataFrame([{
        "Type": data.Type,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        status = "FAILURE"
    else:
        status = "NORMAL"

    return {
        "model_version": model_version,
        "machine_status": status,
        "failure_probability": round(float(probability), 4)
    }

@app.post("/feedback")
def submit_feedback(data: FeedbackData):

    feedback_file = "data/feedback.csv"

    feedback_row = pd.DataFrame([{
        "Type": data.Type,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear,
        "prediction": data.prediction,
        "actual": data.actual,
        "model_version": data.model_version,
        "timestamp": datetime.now().isoformat()
    }])

    if os.path.exists(feedback_file):
        feedback_row.to_csv(
            feedback_file,
            mode="a",
            header=False,
            index=False
        )
    else:
        feedback_row.to_csv(
            feedback_file,
            index=False
        )

    return {
        "message": "Feedback recorded successfully",
        "model_version": data.model_version
    }

@app.get("/monitor")
def get_monitoring_data():

    feedback_file = "data/feedback.csv"

    df = pd.read_csv(feedback_file)

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    prediction = df["prediction"]
    actual = df["actual"]

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

    if recall >= 0.70:
        health = "HEALTHY"
        retraining = "NOT REQUIRED"
    else:
        health = "DEGRADED"
        retraining = "REQUIRED"

    with open(ACTIVE_MODEL_FILE, "r") as file:
        model_version = file.read().strip()

    return {
        "model_version": model_version,
        "total_predictions": len(df),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "health": health,
        "retraining": retraining
    }