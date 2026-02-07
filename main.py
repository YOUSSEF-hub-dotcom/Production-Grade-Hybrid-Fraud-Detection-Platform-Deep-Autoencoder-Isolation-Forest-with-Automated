import logging
import argparse
from logger_config import setup_logging
from data import load_and_preprocess_data
from model import build_and_train_hybrid_model
from MLflow_LifeCycle import run_mlflow_lifecycle

# 1. إعداد الـ Logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    # إعداد الـ Argument Parser لاستقبال القيم من MLproject
    parser = argparse.ArgumentParser(description="Credit Card Fraud Hybrid Detection Pipeline")
    
    parser.add_argument("--mse_threshold_pct", type=float, default=95.0, help="Percentile for AE MSE threshold")
    parser.add_argument("--iso_threshold_pct", type=float, default=3.0, help="Percentile for ISO Forest score threshold")
    parser.add_argument("--outlier_fraction", type=float, default=0.05, help="Contamination factor for ISO Forest training")
    
    args, unknown = parser.parse_known_args()

    logger.info("🚀 Starting Credit Card Fraud Detection Pipeline...")
    logger.info(f"Parameters: MSE_Pct={args.mse_threshold_pct}, ISO_Pct={args.iso_threshold_pct}, Outlier={args.outlier_fraction}")

    try:
        # 2. مرحلة البيانات (Data Pipeline)
        # ======================================================
        logger.info("Step 1: Data Preprocessing & EDA")
        df = load_and_preprocess_data(r"creditcard.csv")

        # 3. مرحلة التدريب (Model Training)
        # نمرر الباراميترز اللي استقبلناها للدالة
        # ======================================================
        logger.info("Step 2: Training Hybrid AE-ISO Forest Model")
        model_results = build_and_train_hybrid_model(
            df, 
            mse_threshold_pct=args.mse_threshold_pct,
            iso_threshold_pct=args.iso_threshold_pct,
            outlier_fraction=args.outlier_fraction
        )
        
        # استخراج النتائج
        X_test = model_results["X_test"]
        y_test = model_results["y_test"]
        scaler = model_results["scaler"]
        autoencoder = model_results["autoencoder"]
        encoder = model_results["encoder"]
        iso_forest = model_results["iso_forest"]
        mse_threshold = model_results["mse_threshold"]
        iso_threshold = model_results["iso_threshold"]

        # 4. مرحلة الـ MLOps (MLflow Lifecycle)
        # نرسل القيم المحسوبة والنسب الأصلية للتسجيل (Logging)
        # ======================================================
        logger.info("Step 3: Logging Model to MLflow & Model Registry")
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
            mse_threshold_pct=args.mse_threshold_pct,    # نرسل النسبة للتسجيل
            iso_threshold_pct=args.iso_threshold_pct,    # نرسل النسبة للتسجيل
            outlier_fraction=args.outlier_fraction       # نرسل النسبة للتسجيل
        )

        logger.info(f"✅ Pipeline Completed Successfully! Run ID: {run_id}")

    except Exception as e:
        logger.error(f"❌ An error occurred during the pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    main()