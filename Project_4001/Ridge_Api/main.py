from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

artifacts = joblib.load("ridge_trp_model.joblib")
model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]  #['EXR','OILP','PMI','TDC','TRI','TRPD','TRS','TRX','WRP']

class PredictRequest(BaseModel):
    EXR: float
    OILP: float
    PMI: float
    TDC: float
    TRI: float
    TRPD: float
    TRS: float
    TRX: float
    WRP: float

class PredictResponse(BaseModel):
    trp_pred: float

app = FastAPI(
    title="Thai Rubber Price Forecast (Ridge Regression)",
    description="MLaaS API for TRP using Optuna-tuned Ridge + external factors",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"message": "TRP Ridge Regression API is running."}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    x = np.array([[getattr(request, f) for f in features]])

    x_scaled = scaler.transform(x)

    #Predict
    y_pred = model.predict(x_scaled)[0]

    return PredictResponse(trp_pred=float(y_pred))
