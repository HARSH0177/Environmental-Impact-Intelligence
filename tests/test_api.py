"""
Unit and Integration Tests for Environmental Impact Intelligence API
Author: Harsh Ambule (github.com/HARSH0177)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "mobilenet_v2_deforestation" in data["models_loaded"]
    assert "xgboost_pm25_regressor" in data["models_loaded"]


def test_pm25_forecast_endpoint():
    payload = {
        "city": "Delhi",
        "pm25_lag_1": 120.5,
        "pm25_lag_2": 115.0,
        "pm25_lag_3": 110.0,
        "pm25_lag_7": 98.0,
        "pm25_lag_14": 92.0,
        "pm10": 210.0,
        "month": 11  # Winter regime
    }
    response = client.post("/api/v1/forecast/pm25", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["city"] == "Delhi"
    assert data["predicted_pm25_ug_m3"] > 0
    assert "confidence_interval_90" in data
    assert data["confidence_interval_90"]["lower"] < data["confidence_interval_90"]["upper"]
    assert data["aqi_category"] in ["Poor", "Very Poor", "Severe"]
    assert "feature_contributions_shap" in data


def test_deforestation_classifier_intact_forest():
    payload = {
        "latitude": 21.1458,
        "longitude": 79.0882,
        "ndvi_value": 0.65  # High vegetation canopy
    }
    response = client.post("/api/v1/classify/deforestation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["alert_level"] == "NORMAL"
    assert data["deforestation_probability"] < 0.30
    assert "Dense Intact Canopy" in data["canopy_status"]


def test_deforestation_classifier_critical_clearance():
    payload = {
        "latitude": 21.1458,
        "longitude": 79.0882,
        "ndvi_value": 0.12  # Severely stripped vegetation
    }
    response = client.post("/api/v1/classify/deforestation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["alert_level"] == "CRITICAL"
    assert data["deforestation_probability"] > 0.70
    assert "Active Deforestation" in data["canopy_status"]


def test_city_climate_risk_registry():
    # Test registered city
    resp_delhi = client.get("/api/v1/city-climate-risk/delhi")
    assert resp_delhi.status_code == 200
    delhi_data = resp_delhi.json()
    assert delhi_data["found"] is True
    assert delhi_data["pm25_p50_ug_m3"] == 108.4

    # Test unregistered city fallback
    resp_unknown = client.get("/api/v1/city-climate-risk/chandigarh")
    assert resp_unknown.status_code == 200
    unknown_data = resp_unknown.json()
    assert unknown_data["found"] is False
    assert "pm25_p50" in unknown_data
