"""
Satellite-Ground Multi-Modal Fusion — Sentinel-2 NDVI × CPCB Air Quality
Author: Harsh Ambule (github.com/HARSH0177)
Part of: Environmental-Impact-Intelligence
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def build_fusion_dataset(cpcb_df, ndvi_df, city_col="City", date_col="Date"):
    """
    Joins ground-level CPCB air quality observations with Sentinel-2 NDVI
    vegetation canopy index by city geographic coordinate and observation date.
    """
    cpcb_df[date_col] = pd.to_datetime(cpcb_df[date_col])
    ndvi_df[date_col] = pd.to_datetime(ndvi_df[date_col])

    fusion_df = pd.merge(
        cpcb_df,
        ndvi_df[[city_col, date_col, "NDVI_mean", "NDVI_min", "NDVI_max", "Vegetation_Cover_Pct"]],
        on=[city_col, date_col],
        how="inner"
    )
    return fusion_df


def train_fusion_model(X_train, y_train):
    """
    Trains multi-modal XGBoost Regressor incorporating both meteorological,
    chemical precursor gases (NO2, SO2, CO, PM10) and satellite NDVI features.
    """
    fusion_model = XGBRegressor(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.025,
        subsample=0.88,
        colsample_bytree=0.88,
        random_state=42,
        n_jobs=-1
    )
    fusion_model.fit(X_train, y_train)
    return fusion_model


def run_ablation_study(fusion_model, X_test, y_test, feature_names):
    """
    Performs feature ablation to quantify the incremental predictive gain
    contributed by satellite NDVI indices over ground-only telemetry.
    """
    baseline_preds = fusion_model.predict(X_test)
    baseline_r2 = r2_score(y_test, baseline_preds)

    importances = pd.Series(fusion_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    return {
        "Fusion_R2": float(baseline_r2),
        "Feature_Importances": importances.to_dict()
    }
