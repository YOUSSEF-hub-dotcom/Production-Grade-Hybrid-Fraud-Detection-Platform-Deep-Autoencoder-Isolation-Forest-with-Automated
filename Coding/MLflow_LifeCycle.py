import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import joblib
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
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
    Consistently handles preprocessing (Log Transform & Scaling) during inference.
    """
    def __init__(self, feature_columns, mse_threshold, iso_threshold):
        self.feature_columns = feature_columns
        self.mse_threshold = mse_threshold
        self.iso_threshold = iso_threshold

    def load_context(self, context):
        # Load serialized scikit-learn/joblib components
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.iso_forest = joblib.load(context.artifacts["iso_forest"])
        
        # FIXED: Load native TensorFlow/Keras models securely using Keras native loader
        self.autoencoder = tf.keras.models.load_model(context.artifacts["autoencoder"])
        self.encoder = tf.keras.models.load_model(context.artifacts["encoder"])

    def predict(self, context, model_input):
        # Ensure input is structural DataFrame
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input, columns=self.feature_columns)

        # Explicit copy to safeguard runtime data modifications
        X = model_input[self.feature_columns].copy()
        
        # FIXED: Apply Feature Consistency Rule (Log Transformation for Skewed Amount)
        if 'Amount' in X.columns:
            X['Amount'] = np.log1p(X['Amount'])

        # Preprocessing: Apply the isolated Train Scaler Template
        X_scaled = self.scaler.transform(X)

        # FIXED: Convert Keras Tensor to Numpy Array (.numpy()) to prevent scikit-learn crash
        latent_tensor = self.encoder(X_scaled, training=False)
        latent_numpy = latent_tensor.numpy()
        
        # Step 1: Isolation Forest detection via Latent Space
        iso_scores = self.iso_forest.decision_function(latent_numpy)
        iso_pred = (iso_scores <= self.iso_threshold).astype(int)

        # Step 2: Autoencoder detection via Reconstruction Error (MSE)
        recon_tensor = self.autoencoder(X_scaled, training=False)
        mse = np.mean((X_scaled - recon_tensor.numpy()) ** 2, axis=1)
        ae_pred = (mse > self.mse_threshold).astype(int)

        # Step 3: Hybrid Logic (OR gate) – High Recall Production Strategy
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
    Manages the complete MLflow lifecycle: Logging features, metrics, native artifacts, 
    and enforcing strict Automated Quality Gates for Production deployment.
    """
    logger.info("--- Initializing MLflow Lifecycle for Fraud Detection ---")

    EXPERIMENT_NAME = "Credit_Card_Fraud_Hybrid"
    MODEL_NAME = "Credit_Fraud_Hybrid_Model"

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # FIXED: Serialize models natively according to framework standards (Joblib vs Keras Native)
    logger.info("Serializing model components securely...")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(iso_forest, "iso_forest.pkl")
    autoencoder.save("autoencoder.keras")
    encoder.save("encoder.keras")

    artifacts = {
        "scaler": "scaler.pkl",
        "iso_forest": "iso_forest.pkl",
        "autoencoder": "autoencoder.keras",
        "encoder": "encoder.keras"
    }

    with mlflow.start_run(run_name=f"Recall_Run_ISO_{iso_threshold_pct}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run successfully started. ID: {run_id}")

        # 1. Log Training Configuration & Hyperparameters
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

        # 2. Complete Model Evaluation Logic on Test Split
        logger.info("Performing pipeline evaluation on test set...")
        
        # Fixed: Apply identical post-firewall treatment to evaluation mirror
        X_eval = X_test[feature_columns].copy()
        if 'Amount' in X_eval.columns:
            X_eval['Amount'] = np.log1p(X_eval['Amount'])
            
        X_scaled = scaler.transform(X_eval)
        latent = encoder.predict(X_scaled)
        
        # Calculate Isolation Forest scores
        iso_scores = iso_forest.decision_function(latent)
        iso_pred = (iso_scores <= iso_threshold).astype(int)

        # Calculate Autoencoder Reconstruction Error
        recon = autoencoder.predict(X_scaled)
        mse = np.mean((X_scaled - recon) ** 2, axis=1)
        ae_pred = (mse > mse_threshold).astype(int)

        # Final Hybrid Prediction Matrix
        final_pred = ((iso_pred == 1) | (ae_pred == 1)).astype(int)

        # 3. Compute and Log System Metrics
        logger.info("Calculating performance metrics...")
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
        logger.info("Generating and logging analytical artifacts...")
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

        # 5. Model Signature Generation via Temporary Wrapper Context
        logger.info("Inferring complex model schema signature...")
        input_example = X_test[feature_columns].iloc[:5]
        
        temp_wrapper = CreditFraudWrapper(feature_columns, mse_threshold, iso_threshold)
        temp_wrapper.scaler = scaler
        temp_wrapper.autoencoder = autoencoder
        temp_wrapper.encoder = encoder
        temp_wrapper.iso_forest = iso_forest
        
        output_example = temp_wrapper.predict(None, input_example)
        signature = infer_signature(input_example, output_example)

        # Log PyFunc Model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CreditFraudWrapper(feature_columns, mse_threshold, iso_threshold),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example
        )

        # 6. Model Registry & Automatic Versioning Control
        mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME)
        latest_v_info = client.get_latest_versions(MODEL_NAME, stages=["None"])
        latest_version = latest_v_info[0].version

        # Transition Version to Staging Stage
        client.transition_model_version_stage(
            name=MODEL_NAME, version=latest_version, stage="Staging"
        )

        # 7. Production Quality Gate Enforcement (Rule-Based Gates)
        logger.info(f"Enforcing Quality Gate for Registry Version {latest_version}...")
        if recall_fraud >= 0.80:
            client.transition_model_version_stage(
                name=MODEL_NAME, version=latest_version, stage="Production", archive_existing_versions=True
            )
            logger.info(f" SUCCESS: Model v{latest_version} promoted to Production Environment | Recall: {recall_fraud:.2%}")
        else:
            logger.warning(f" FAILED: Model v{latest_version} rejected from Production | Recall: {recall_fraud:.2%}")

        # Clean up temporary local files after successful lifecycle run
        for file in ["scaler.pkl", "iso_forest.pkl", "autoencoder.keras", "encoder.keras", "confusion_matrix.png"]:
            if os.path.exists(file):
                os.remove(file)

        return run_id
