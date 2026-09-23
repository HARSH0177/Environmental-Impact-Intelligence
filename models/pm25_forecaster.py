"""
PM2.5 Temporal Forecaster — Gradient Boosted Lag Modeling on CPCB Sensor Records
Author: Harsh Ambule (github.com/HARSH0177)
Part of: Environmental-Impact-Intelligence
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def generate_lag_features(df, target_col="PM2.5", lags=[1, 2, 3, 7, 14], roll_windows=[3, 7]):
    """
    Generates time-series lag and rolling statistics for air quality telemetry.
    """
    data = df.copy()
    data = data.sort_values(by="Date").reset_index(drop=True)

    for lag in lags:
        data[f"{target_col}_lag_{lag}"] = data[target_col].shift(lag)

    for window in roll_windows:
        data[f"{target_col}_roll_mean_{window}"] = data[target_col].shift(1).rolling(window=window).mean()
        data[f"{target_col}_roll_std_{window}"] = data[target_col].shift(1).rolling(window=window).std()

    data["Month"] = pd.to_datetime(data["Date"]).dt.month
    data["DayOfWeek"] = pd.to_datetime(data["Date"]).dt.dayofweek
    data["DayOfYear"] = pd.to_datetime(data["Date"]).dt.dayofyear

    data = data.dropna().reset_index(drop=True)
    return data


def train_pm25_forecaster(X_train, y_train, n_estimators=300, max_depth=6, learning_rate=0.03):
    """
    Trains an XGBoost Regressor for PM2.5 forecasting with regularized trees.
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_forecaster(model, X_test, y_test):
    """
    Computes regression evaluation benchmarks (R2, MAE, RMSE).
    """
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return {
        "R2_Score": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Predictions": preds
    }
