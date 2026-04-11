import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    recall_score,
    precision_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from logger_config import setup_logging

# Initialize Logging
setup_logging()
logger = logging.getLogger("MLflow")

# ======================================================
# PyFunc Wrapper – Hybrid Fraud Model
# ======================================================

class CreditFraudWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow wrapper for the Hybrid Fraud Detection system.
    Combines Autoencoder Reconstruction Error and Isolation Forest Anomaly Scores.
    """
    def __init__(self, feature_columns, mse_threshold, iso_threshold):
        self.feature_columns = feature_columns
        self.mse_threshold = mse_threshold
        self.iso_threshold = iso_threshold

    def load_context(self, context):
        # Load serialized components from MLflow artifacts
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.autoencoder = joblib.load(context.artifacts["autoencoder"])
        self.encoder = joblib.load(context.artifacts["encoder"])
        self.iso_forest = joblib.load(context.artifacts["iso_forest"])

    def predict(self, context, model_input):
        # Ensure input is a DataFrame
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input, columns=self.feature_columns)

        # Preprocessing: Scale features
        X = model_input[self.feature_columns]
        X_scaled = self.scaler.transform(X)

        # Step 1: Isolation Forest detection via Encoder Latent Space
        latent = self.encoder(X_scaled, training=False)
        iso_scores = self.iso_forest.decision_function(latent)
        iso_pred = (iso_scores <= self.iso_threshold).astype(int)

        # Step 2: Autoencoder detection via Reconstruction Error (MSE)
        recon = self.autoencoder(X_scaled, training=False)
        mse = np.mean((X_scaled - recon.numpy()) ** 2, axis=1)
        ae_pred = (mse > self.mse_threshold).astype(int)

        # Step 3: Hybrid Logic (OR gate) – High Recall Strategy
        final_pred = ((iso_pred == 1) | (ae_pred == 1)).astype(int)

        return pd.DataFrame({
            "fraud_prediction": final_pred,
            "iso_score": iso_scores,
            "reconstruction_error": mse
        })

# ======================================================
# MLflow Lifecycle – Dynamic Parameters Integration
# ======================================================

def run_mlflow_lifecycle(
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_columns: list,
        scaler,
        autoencoder,
        encoder,
        iso_forest,
        mse_threshold: float,
        iso_threshold: float,
        mse_threshold_pct: float,
        iso_threshold_pct: float,
        outlier_fraction: float
):
    """
    Manages the MLflow lifecycle: Logging params, metrics, artifacts, 
    and enforcing Quality Gates for Production.
    """
    logger.info("--- Initializing MLflow Lifecycle for Fraud Detection ---")

    EXPERIMENT_NAME = "Credit_Card_Fraud_Hybrid"
    MODEL_NAME = "Credit_Fraud_Hybrid_Model"

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # Serialize artifacts for logging
    logger.info("Serializing model components (Scaler, Encoder, Autoencoder, IsoForest)...")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(autoencoder, "autoencoder.pkl")
    joblib.dump(encoder, "encoder.pkl")
    joblib.dump(iso_forest, "iso_forest.pkl")

    artifacts = {
        "scaler": "scaler.pkl",
        "autoencoder": "autoencoder.pkl",
        "encoder": "encoder.pkl",
        "iso_forest": "iso_forest.pkl"
    }

    with mlflow.start_run(run_name=f"Recall_Run_ISO_{iso_threshold_pct}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run successfully started. ID: {run_id}")

        # 1. Log Training Parameters
        logger.info("Logging model hyperparameters and dynamic thresholds...")
        mlflow.log_params({
            "latent_dim": encoder.output_shape[1],
            "iso_estimators": iso_forest.n_estimators,
            "outlier_fraction_input": outlier_fraction,
            "mse_threshold_percentile": mse_threshold_pct,
            "iso_threshold_percentile": iso_threshold_pct,
            "actual_mse_threshold": mse_threshold,
            "actual_iso_threshold": iso_threshold,
            "optimization_target": "recall_fraud"
        })

        # 2. Model Evaluation Logic
        logger.info("Performing model evaluation on test set...")
        X_scaled = scaler.transform(X_test)
        latent = encoder.predict(X_scaled)
        
        # Calculate Isolation Forest scores
        iso_scores = iso_forest.decision_function(latent)
        iso_pred = (iso_scores <= iso_threshold).astype(int)

        # Calculate Autoencoder Reconstruction Error
        recon = autoencoder.predict(X_scaled)
        mse = np.mean((X_scaled - recon) ** 2, axis=1)
        ae_pred = (mse > mse_threshold).astype(int)

        # Final Hybrid Prediction
        final_pred = ((iso_pred == 1) | (ae_pred == 1)).astype(int)

        # 3. Calculate and Log Metrics
        logger.info("Calculating performance metrics (Recall, Precision, Accuracy)...")
        recall_fraud = recall_score(y_test, final_pred, pos_label=1)
        precision_fraud = precision_score(y_test, final_pred, pos_label=1)
        accuracy = accuracy_score(y_test, final_pred)

        mlflow.log_metrics({
            "recall_fraud": recall_fraud,
            "precision_fraud": precision_fraud,
            "accuracy": accuracy,
            "false_negative_rate": 1 - recall_fraud,
            "mean_iso_score": np.mean(iso_scores),
            "stability_iso_score": np.std(iso_scores)
        })

        # 4. Artifact Logging (Reports & Visualizations)
        logger.info("Generating and logging artifacts (Confusion Matrix, Classification Report)...")
        report = classification_report(y_test, final_pred, output_dict=True)
        mlflow.log_dict(report, "classification_report.json")

        cm = confusion_matrix(y_test, final_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
        plt.title("Hybrid Fraud Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # 5. Model Signature & Registration
        logger.info("Informing model signature and registering Hybrid Model...")
        input_example = X_test[feature_columns].iloc[:5]
        
        # Temp wrapper to infer signature
        temp_wrapper = CreditFraudWrapper(feature_columns, mse_threshold, iso_threshold)
        temp_wrapper.scaler = scaler
        temp_wrapper.autoencoder = autoencoder
        temp_wrapper.encoder = encoder
        temp_wrapper.iso_forest = iso_forest
        
        output_example = temp_wrapper.predict(None, input_example)
        signature = infer_signature(input_example, output_example)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CreditFraudWrapper(feature_columns, mse_threshold, iso_threshold),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example
        )

        # 6. Model Registry & Stage Transition
        mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME)
        latest_v_info = client.get_latest_versions(MODEL_NAME, stages=["None"])
        latest_version = latest_v_info[0].version

        # Transition to Staging
        client.transition_model_version_stage(
            name=MODEL_NAME, version=latest_version, stage="Staging"
        )

        # 7. Production Quality Gate
        logger.info(f"Enforcing Quality Gate for Version {latest_version}...")
        if recall_fraud >= 0.80:
            client.transition_model_version_stage(
                name=MODEL_NAME, version=latest_version, stage="Production", archive_existing_versions=True
            )
            logger.info(f"🚀 SUCCESS: Model v{latest_version} promoted to Production | Recall: {recall_fraud:.2%}")
        else:
            logger.warning(f"❌ FAILED: Model v{latest_version} does not meet Production criteria | Recall: {recall_fraud:.2%}")

        return run_id
