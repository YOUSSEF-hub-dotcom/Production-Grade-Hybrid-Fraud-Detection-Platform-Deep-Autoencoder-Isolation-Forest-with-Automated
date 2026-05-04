import logging
import argparse
from logger_config import setup_logging
from data import load_and_preprocess_data
from model import build_and_train_hybrid_model
from MLflow_LifeCycle import run_mlflow_lifecycle

# 1. Logging Configuration
setup_logging()
logger = logging.getLogger("main")

def main():
    """
    Main Orchestrator for the Credit Card Fraud Detection Pipeline.
    Handles Data Ingestion, Hybrid Model Training, and MLOps Lifecycle.
    """
    # Initialize Argument Parser to receive values from MLproject or CLI
    parser = argparse.ArgumentParser(description="Credit Card Fraud Hybrid Detection Pipeline")
    
    # Dynamic Hyperparameters for Thresholding and Anomaly Detection
    parser.add_argument("--mse_threshold_pct", type=float, default=95.0, help="Percentile for AE MSE threshold")
    parser.add_argument("--iso_threshold_pct", type=float, default=3.0, help="Percentile for ISO Forest score threshold")
    parser.add_argument("--outlier_fraction", type=float, default=0.05, help="Contamination factor for ISO Forest training")
    
    args, unknown = parser.parse_known_args()

    logger.info("🚀 Starting Credit Card Fraud Detection Pipeline Execution...")
    logger.info(f"Execution Parameters: MSE_Pct={args.mse_threshold_pct}, ISO_Pct={args.iso_threshold_pct}, Outlier={args.outlier_fraction}")

    try:
        # 2. Data Engineering Stage
        # ======================================================
        logger.info("Step 1: Ingesting Data & Executing Preprocessing Pipeline...")
        df = load_and_preprocess_data(r"creditcard.csv")

        # 3. Model Training Stage (Hybrid AE-ISO)
        # Passing the parsed arguments to the training function
        # ======================================================
        logger.info("Step 2: Building and Training the Hybrid Autoencoder-Isolation Forest Model...")
        model_results = build_and_train_hybrid_model(
            df, 
            mse_threshold_pct=args.mse_threshold_pct,
            iso_threshold_pct=args.iso_threshold_pct,
            outlier_fraction=args.outlier_fraction
        )
        
        # Extracting trained components and computed thresholds
        X_test = model_results["X_test"]
        y_test = model_results["y_test"]
        scaler = model_results["scaler"]
        autoencoder = model_results["autoencoder"]
        encoder = model_results["encoder"]
        iso_forest = model_results["iso_forest"]
        mse_threshold = model_results["mse_threshold"]
        iso_threshold = model_results["iso_threshold"]

        # 4. MLOps Stage (MLflow Lifecycle & Governance)
        # Logging metrics, artifacts, and registering the model with dynamic thresholds
        # ======================================================
        logger.info("Step 3: Managing Model Lifecycle (Tracking, Artifacts, & Registry)...")
        run_id = run_mlflow_lifecycle(
            X_test=X_test,
            y_test=y_test,
            feature_columns=X_test.columns.tolist(),
            scaler=scaler,
            autoencoder=autoencoder,
            encoder=encoder,
            iso_forest=iso_forest,
            mse_threshold=mse_threshold,
            iso_threshold=iso_threshold,
            mse_threshold_pct=args.mse_threshold_pct,    # Log the original percentile
            iso_threshold_pct=args.iso_threshold_pct,    # Log the original percentile
            outlier_fraction=args.outlier_fraction       # Log the contamination factor
        )

        logger.info(f" Pipeline Completed Successfully! MLflow Run ID: {run_id}")

    except Exception as e:
        logger.error(f" Pipeline Execution Failed: {str(e)}")
        # Re-raise to signal failure to MLflow or CI/CD runner
        raise e

if __name__ == "__main__":
    main()
