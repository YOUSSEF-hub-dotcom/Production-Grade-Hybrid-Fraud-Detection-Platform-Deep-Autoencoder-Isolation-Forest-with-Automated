🛡️ Deep-Hybrid Fraud Detection System
Integrating Autoencoder with Isolation Forest

🎯 Project Goal

The goal of this project is to design and implement a production-grade hybrid fraud detection system capable of identifying unknown (zero-day) fraudulent transactions using an unsupervised learning approach.

Unlike traditional rule-based or supervised systems, this solution:

Learns normal transaction behavior
Detects deviations as anomalies
Combines deep learning (Autoencoder) with statistical isolation (Isolation Forest)
Operates in real-time with MLOps integration

📌 Overview

This project presents a state-of-the-art hybrid fraud detection system that combines:

🧠 Autoencoder (Deep Learning) → Behavioral anomaly detection
🌲 Isolation Forest (Machine Learning) → Geometric anomaly detection

The system is designed to detect credit card fraud in real-time, even for previously unseen attack patterns (Zero-Day Fraud).

🚀 Key Features
✅ Hybrid Detection (AE + Isolation Forest)
✅ Unsupervised Learning (No dependency on labels)
✅ Real-time inference via API
✅ Full MLOps pipeline with MLflow
✅ Model Registry + Quality Gates
✅ Interactive Dashboard (Streamlit)
✅ Enterprise-grade API (FastAPI)
🧠 Why Hybrid Model?

Instead of relying on a single algorithm:

Component	Role
Autoencoder	Learns "normal behavior" & detects reconstruction anomalies
Isolation Forest	Detects outliers in latent feature space
Hybrid Logic	Flags fraud if ANY model detects anomaly
final_prediction = 1 if (iso == 1 or ae == 1) else 0

💡 Philosophy:

"If either model is suspicious → treat as fraud (Safety First)"

🏗️ Architecture
Raw Data (30 Features)
        ↓
[Autoencoder]
   → Latent Space (10 Features)
   → Reconstruction Error (MSE)
        ↓
[Isolation Forest]
   → Anomaly Score
        ↓
[Hybrid Decision Layer]
   → Final Fraud Prediction
📊 Dataset
Source: Credit Card Transactions Dataset
Records: ~284,807
Features:
28 PCA-transformed features (V1–V28)
Time
Amount
Target:
0 → Normal
1 → Fraud (used only for evaluation)
🔍 Data Processing
✔️ Cleaning
Removed 1,081 duplicate records
No missing values
✔️ Feature Engineering
Log Transformation:
df['Amount'] = np.log1p(df['Amount'])
✔️ Important Decision
❗ Outliers were NOT removed

لأنهم يمثلوا حالات fraud فعليًا

📈 Exploratory Data Analysis (EDA)

Key insights:

⚠️ Extreme class imbalance
💰 Fraud mimics normal transaction amounts
⏰ Fraud often occurs at unusual times
🔬 t-SNE shows fraud forms distinct patterns
🧠 Model Details
🔹 Autoencoder
Architecture:
30 → 128 → 64 → 32 → 16 → 10 → 16 → 32 → 64 → 128 → 30
Latent Dimension: 10
Loss: MSE
Optimizer: Adam
Purpose:
Feature compression
Reconstruction-based anomaly detection
🔹 Isolation Forest
Trees: 100
Contamination: configurable
Input: Latent features
🔹 Thresholding
Metric	Strategy
MSE	Top 5% anomalies
ISO Score	Bottom 3% anomalies
⚙️ Pipeline Flow
Data Preprocessing (data.py)
Model Training (model.py)
Hybrid Evaluation
MLflow Logging (mlflow_lifeCycle.py)
Deployment (API + Dashboard)
📊 Results
Metric	Value
🎯 Recall (Fraud)	83.33%
⚠️ Precision	2.03%
✅ Accuracy	93.59%
❌ False Negative Rate	16.67%
⚖️ Important Insight

Low precision is acceptable in fraud detection

Why?

Fraud cases are extremely rare (<0.2%)
Missing fraud = high financial loss 💰
False positives = minor inconvenience
🔄 MLOps (MLflow)
✔️ Features
Experiment tracking
Model versioning
Artifact storage
Automated quality gates
✔️ Quality Gate
if recall_fraud >= 0.80:
    → Production
else:
    → Rejected
📦 Model Artifacts
scaler.pkl
autoencoder.pkl
encoder.pkl
iso_forest.pkl
🌐 API (FastAPI)
Endpoint
POST /predict
Input
{
  "features": [30 values]
}
Output
{
  "fraud_prediction": 1,
  "iso_score": -0.03,
  "reconstruction_error": 0.45
}
Features
JWT-based rate limiting
Logging to database
Request tracking (UUID)
Performance monitoring
📊 Dashboard (Streamlit)

Features:

🔍 Real-time fraud detection
📈 Visualization (MSE + ISO Score)
🕸️ Radar chart for features
📊 Historical fraud analysis
▶️ How to Run
1. Install dependencies
conda env create -f conda.yaml
conda activate fraud_detection_env
2. Run ML Pipeline
mlflow run . \
  -P mse_threshold_pct=95 \
  -P iso_threshold_pct=3 \
  -P outlier_fraction=0.05
3. Run API
python api.py
4. Run Dashboard
streamlit run app.py
🔮 Future Improvements
🔄 Drift Detection & Auto-Retraining
📊 Explainable AI (SHAP)
🌐 Multi-region models
⚡ Streaming with Kafka
🤖 Graph-based fraud detection
💼 Business Impact
💰 Reduce fraud losses significantly
⚡ Real-time decision making
🔒 Detect unknown fraud patterns
📈 Scalable enterprise solution
👨‍💻 Author

[Youssef Mahmoud

⭐ Final Note

This is not just a model — it's a complete AI-powered fraud detection platform ready for real-world deployment.
