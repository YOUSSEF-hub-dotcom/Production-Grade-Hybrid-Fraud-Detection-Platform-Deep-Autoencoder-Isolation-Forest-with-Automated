# 🛡️ Deep-Hybrid Fraud Detection System  
### Integrating Autoencoder with Isolation Forest

---

## 🎯 Project Goal

Design and implement a **production-grade hybrid fraud detection system** capable of identifying **zero-day fraudulent transactions** using **Unsupervised Learning**.

### 🚨 Unlike traditional systems:
- No dependency on fixed rules  
- No dependency on labeled data  
- Learns **normal behavior** and detects deviations  

---

## 📌 Overview

This system combines:

- 🧠 **Autoencoder (Deep Learning)** → Behavioral Anomaly Detection  
- 🌲 **Isolation Forest (Machine Learning)** → Geometric Outlier Detection  

📍 Goal: Detect fraud even for **unseen patterns (Zero-Day Attacks)**

---

## 🚀 Key Features

- ✅ Hybrid Detection (Autoencoder + Isolation Forest)  
- ✅ Fully Unsupervised  
- ✅ Real-time Inference API  
- ✅ MLflow MLOps Pipeline  
- ✅ Model Registry + Quality Gates  
- ✅ Streamlit Dashboard  
- ✅ Enterprise-grade FastAPI  

---

## 🧠 Why Hybrid Model?

| Component | Role |
|----------|------|
| Autoencoder | Learns normal behavior and detects anomalies |
| Isolation Forest | Detects outliers in latent space |
| Hybrid Logic | Flags fraud if ANY model detects anomaly |

```python
final_prediction = 1 if (iso == 1 or ae == 1) else 0
```

### 💡 Philosophy:
> "If either model is suspicious → treat as fraud (Safety First)"

---

## 🏗️ System Architecture

```
Raw Data (30 Features)
        ↓
   [Autoencoder]
        ↓
 Latent Space (10)
        ↓
 Reconstruction Error (MSE)
        ↓
  [Isolation Forest]
        ↓
   Anomaly Score
        ↓
[Hybrid Decision Layer]
        ↓
 Final Fraud Prediction
```

---

## 📊 Dataset

- Source: Credit Card Transactions Dataset  
- Records: ~284,807  

### Features:
- 28 PCA Features (V1–V28)  
- Time  
- Amount  

### Target:
- `0` → Normal  
- `1` → Fraud (used only for evaluation)  

---

## 🔍 Data Processing

### ✔️ Cleaning
- Removed 1,081 duplicate records  
- No missing values  

### ✔️ Feature Engineering

```python
df['Amount'] = np.log1p(df['Amount'])
```

### ❗ Important Decision
Outliers were NOT removed  
Because they represent real fraud cases  

---

## 📈 Exploratory Data Analysis (EDA)

### Key Insights:

- ⚠️ Extreme class imbalance  
- 💰 Fraud mimics normal transactions  
- ⏰ Fraud occurs at unusual times  
- 🔬 t-SNE shows distinct fraud clusters  

---

## 🧠 Model Details

### 🔹 Autoencoder

Architecture:
```
30 → 128 → 64 → 32 → 16 → 10 → 16 → 32 → 64 → 128 → 30
```

- Latent Dimension: 10  
- Loss: MSE  
- Optimizer: Adam  

Purpose:
- Feature Compression  
- Reconstruction-based Detection  

---

### 🔹 Isolation Forest

- Trees: 100  
- Input: Latent Features  
- Contamination: configurable  

---

### 🔹 Thresholding

| Metric | Strategy |
|--------|--------|
| MSE | Top 5% anomalies |
| ISO Score | Bottom 3% anomalies |

---

## ⚙️ Pipeline Flow

1. Data Preprocessing (data.py)  
2. Model Training (model.py)  
3. Hybrid Evaluation  
4. MLflow Logging (mlflow_lifeCycle.py)  
5. Deployment (API + Dashboard)  

---

## 📊 Results

| Metric | Value |
|------|------|
| 🎯 Recall (Fraud) | **83.33%** |
| ⚠️ Precision | **2.03%** |
| ✅ Accuracy | **93.59%** |
| ❌ False Negative Rate | **16.67%** |

---

## ⚖️ Important Insight

Low precision is acceptable in fraud detection

### Why?

- Fraud cases are extremely rare (<0.2%)  
- Missing fraud = high financial loss  
- False positives = minor inconvenience  

---

## 🔄 MLOps (MLflow)

### ✔️ Features
- Experiment Tracking  
- Model Versioning  
- Artifact Storage  
- Automated Quality Gates  

### ✔️ Quality Gate

```python
if recall_fraud >= 0.80:
    → Production
else:
    → Rejected
```

---

## 📦 Model Artifacts

- scaler.pkl  
- autoencoder.pkl  
- encoder.pkl  
- iso_forest.pkl  

---

## 🌐 API (FastAPI)

### Endpoint
```
POST /predict
```

### Input
```json
{
  "features": [30 values]
}
```

### Output
```json
{
  "fraud_prediction": 1,
  "iso_score": -0.03,
  "reconstruction_error": 0.45
}
```

### Features
- JWT-based Rate Limiting  
- Logging to Database  
- UUID Request Tracking  
- Performance Monitoring  

---

## 📊 Dashboard (Streamlit)

### Features:

- 🔍 Real-time fraud detection  
- 📈 Visualization (MSE + ISO Score)  
- 🕸️ Radar Charts  
- 📊 Historical Analysis  

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
conda env create -f conda.yaml
conda activate fraud_detection_env
```

### 2. Run ML Pipeline
```bash
mlflow run . \
  -P mse_threshold_pct=95 \
  -P iso_threshold_pct=3 \
  -P outlier_fraction=0.05
```

### 3. Run API
```bash
python api.py
```

### 4. Run Dashboard
```bash
streamlit run app.py
```

---

## 🔮 Future Improvements

- Drift Detection & Auto-Retraining  
- Explainable AI (SHAP)  
- Multi-region Deployment  
- Kafka Streaming  
- Graph-based Fraud Detection  

---

## 💼 Business Impact

- Reduce fraud losses  
- Real-time decision making  
- Detect unknown fraud patterns  
- Scalable enterprise solution  

---

## 👨‍💻 Author

Youssef Mahmoud  

---

## ⭐ Final Note

This is not just a model —  
it's a **complete AI-powered fraud detection platform** ready for real-world deployment 🚀
