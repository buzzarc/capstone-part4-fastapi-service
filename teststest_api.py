import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
mock_model = MagicMock()
mock_model.predict_proba.return_value = [[0.75, 0.25]]
joblib.load = MagicMock(return_value=mock_model)

from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

def test_single_predict_success():
    payload = {
        "customer_id": "CUST00001",
        "recency": 14,
        "frequency": 5,
        "monetary": 150.75,
        "support_tickets": 1,
        "web_events": 42
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert data["customer_id"] == "CUST00001"

def test_predict_validation_error():
    invalid_payload = {
        "customer_id": "CUST00001",
        "recency": -5,
        "frequency": 2,
        "monetary": 20.0,
        "support_tickets": 0,
        "web_events": 10
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422