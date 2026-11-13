import os
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/app/models/model.joblib",
)

model = None


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")

    except Exception as exc:
        print(f"Model loading failed: {type(exc).__name__}")
        model = None

    yield

    model = None


app = FastAPI(
    title="MLOps Assignment 3 - Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "application": "MLOps Inference API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "alive",
    }


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "ready",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is unavailable",
        )

    try:
        prediction = model.predict(
            [request.features]
        )

        return {
            "prediction": int(prediction[0]),
            "model_status": "loaded",
            "note": "Educational demonstration only",
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        )
