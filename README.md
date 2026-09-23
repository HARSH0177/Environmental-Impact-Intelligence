<div align="center">

# 🌍 Environmental-Impact-Intelligence
### **Satellite Deforestation Classification & 5.7M CPCB Ground-Air Quality Fusion**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressors-181717?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Google Earth Engine](https://img.shields.io/badge/Google_Earth_Engine-Sentinel--2-4285F4?style=for-the-badge&logo=googleearth&logoColor=white)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<br>

```text
"End-to-end multi-modal environmental intelligence combining transfer-learning satellite
 deforestation classification with temporal PM2.5 forecasting over 5.7M CPCB sensor records."
```

</div>

---

## 📌 Executive Architecture & Methodological Overview

**Environmental-Impact-Intelligence** provides an end-to-end computational pipeline across three complementary environmental modeling domains:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               PIPELINE ARCHITECTURE                                    │
│                                                                                        │
│  [1. SATELLITE VISION]        [2. GROUND SENSOR TELEMETRY]    [3. MULTI-MODAL FUSION]  │
│  Satellite Land Cover Tiles   5.7M CPCB Sensor Records        Sentinel-2 NDVI Rasters  │
│          │                               │                               │             │
│          ▼                               ▼                               ▼             │
│   MobileNetV2 Transfer           Lag Feature Engineering         Spatiotemporal Join   │
│   Learning (GAP + Dropout)       (Lags 1-14 + Rolling Stats)     Across 29 Indian City │
│          │                               │                               │ Coordinates │
│          ▼                               ▼                               ▼             │
│   Binary Deforestation           XGBoost PM2.5 Forecaster        Joint XGBoost Regressor│
│   Detector (ROC-AUC 0.924)       (R² = 0.902, MAE = 8.42)        (R² = 0.972, Ablation)│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Benchmarks & Performance Metrics

### 1. Satellite Deforestation Classifier (MobileNetV2 Transfer Learning)
Evaluated on balanced validation splits of optical satellite canopy patches:

| Metric | Baseline Linear | Random Forest | **MobileNetV2 (Proposed)** |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.742 | 0.816 | **0.924** |
| **F1-Score** | 0.710 | 0.789 | **0.891** |
| **Accuracy** | 72.4% | 80.1% | **89.6%** |
| **Precision** | 0.695 | 0.820 | **0.884** |
| **Recall** | 0.728 | 0.760 | **0.898** |

### 2. PM2.5 Air Quality Forecasting (5.7M CPCB Sensor Records)
Trained across multi-year Central Pollution Control Board (CPCB) ground monitoring stations:

| Model Architecture | $R^2$ Score | MAE ($\mu g/m^3$) | RMSE ($\mu g/m^3$) |
| :--- | :---: | :---: | :---: |
| Persistence / Naive Baseline | 0.612 | 18.24 | 26.50 |
| Linear Lag Regressor | 0.748 | 13.80 | 19.42 |
| Random Forest Regressor | 0.845 | 10.15 | 14.88 |
| **XGBoost Regressor (Proposed)** | **0.902** | **8.42** | **12.18** |

### 3. Multi-Modal Satellite $\times$ Ground Fusion (Sentinel-2 NDVI Join)
Ablation study over 29 Indian cities joining Sentinel-2 NDVI canopy index with ground telemetry:

| Feature Configuration | $R^2$ Score | MAE ($\mu g/m^3$) | Predictive Gain |
| :--- | :---: | :---: | :---: |
| Ground Chemical Only (PM10, NO2, SO2, CO) | 0.894 | 8.85 | Baseline |
| Ground + Meteorological Lag Features | 0.928 | 6.74 | +3.4% |
| **Ground + Meteorology + Sentinel-2 NDVI Fusion** | **0.972** | **4.15** | **+7.8%** |

---

## 📂 Repository Structure

```text
environmental-impact-intelligence/
├── models/
│   ├── deforestation_classifier.py    # MobileNetV2 transfer learning & fine-tuning
│   ├── pm25_forecaster.py             # Lag feature engineering & XGBoost model
│   └── satellite_ground_fusion.py     # Sentinel-2 NDVI × CPCB fusion pipeline
├── notebooks/
│   └── ENVIRONMENTAL_ANALYSIS_MODEL_FINAL_CLEAN.ipynb  # End-to-end reproducible notebook
├── scripts/
│   └── evaluate.py                    # Benchmark generation & evaluation report
├── requirements.txt                   # Dependency specifications
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start & Reproduction

### 1. Installation
```bash
git clone https://github.com/HARSH0177/Environmental-Impact-Intelligence.git
cd Environmental-Impact-Intelligence
pip install -r requirements.txt
```

### 2. Run Evaluation Benchmarks
```bash
python scripts/evaluate.py
```

### 3. Train Satellite Deforestation Model
```python
from models.deforestation_classifier import build_deforestation_model

model, base_model = build_deforestation_model(input_shape=(224, 224, 3))
model.summary()
```

### 4. Train Multi-Modal Fusion Model
```python
from models.satellite_ground_fusion import build_fusion_dataset, train_fusion_model

fusion_df = build_fusion_dataset(cpcb_df, ndvi_df)
# Train XGBoost Regressor with NDVI features
```

---

## 👤 Author & Research Attribution

Developed and maintained by **Harsh Ambule**:
- **GitHub**: [@HARSH0177](https://github.com/HARSH0177)
- **LinkedIn**: [Harsh Ambule](https://www.linkedin.com/in/harsh-ambule-3551bb266/)
- **Email**: harshambule1129@gmail.com

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
