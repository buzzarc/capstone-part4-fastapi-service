from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
import os

app = FastAPI(
    title="D2C Customer Churn Intelligence & Retention API",
    version="1.0.0"
)

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

class CustomerPayload(BaseModel):
    customer_id: str = Field(..., examples=["CUST00001"])
    recency: int = Field(..., ge=0, examples=[14])
    frequency: int = Field(..., ge=0, examples=[5])
    monetary: float = Field(..., ge=0.0, examples=[150.75])
    support_tickets: int = Field(..., ge=0, examples=[1])
    web_events: int = Field(..., ge=0, examples=[42])

class SinglePredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    predicted_class: int
    risk_explanation: str

class BatchPayload(BaseModel):
    customers: List[CustomerPayload]

class BatchPredictionResponse(BaseModel):
    predictions: List[SinglePredictionResponse]

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model file artifact could not be localized."
        )
    return {"status": "healthy", "model_loaded": True}

def generate_risk_explanation(prob: float, payload: CustomerPayload) -> str:
    if prob > 0.7:
        if payload.support_tickets > 3:
            return "Critical Risk: High probability of churn strongly driven by elevated open support tickets."
        return "High Risk: Long periods of inactivity matching historical churn profiles."
    elif prob > 0.4:
        return "Moderate Risk: Behavioral engagement metrics showing signs of early decay."
    return "Low Risk: User exhibits consistent platform interactions and transactions."

@app.post("/predict", response_model=SinglePredictionResponse, status_code=status.HTTP_200_OK)
def predict_single(payload: CustomerPayload):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is uninitialized."
        )
    try:
        features = np.array([[
            payload.recency,
            payload.frequency,
            payload.monetary,
            payload.support_tickets,
            payload.web_events
        ]])
        prob = float(model.predict_proba(features)[0][1])
        pred_class = int(prob >= 0.42)
        explanation = generate_risk_explanation(prob, payload)
        
        return SinglePredictionResponse(
            customer_id=payload.customer_id,
            churn_probability=prob,
            predicted_class=pred_class,
            risk_explanation=explanation
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/batch_predict", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
def predict_batch(payload: BatchPayload):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is uninitialized."
        )
    try:
        results = []
        for customer in payload.customers:
            features = np.array([[
                customer.recency,
                customer.frequency,
                customer.monetary,
                customer.support_tickets,
                customer.web_events
            ]])
            prob = float(model.predict_proba(features)[0][1])
            pred_class = int(prob >= 0.42)
            explanation = generate_risk_explanation(prob, customer)
            
            results.append(SinglePredictionResponse(
                customer_id=customer.customer_id,
                churn_probability=prob,
                predicted_class=pred_class,
                risk_explanation=explanation
            ))
        return BatchPredictionResponse(predictions=results)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )