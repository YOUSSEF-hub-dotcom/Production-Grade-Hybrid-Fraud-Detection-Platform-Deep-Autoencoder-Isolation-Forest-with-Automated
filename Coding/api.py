import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional
import pandas as pd
import mlflow.pyfunc
import jwt
import joblib

from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from config import settings

# FastAPI application instance
app = FastAPI(title="Enterprise Fraud Detection API")

# Logging Configuration - Setup at the very beginning
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("fraud_api.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AnomalyDetectionAPI")

logger.info("--- Starting Fraud Detection API Service ---")

# Database setup and connectivity
Base = declarative_base()
SessionLocal = None
engine = None

try:
    # Adding pooling configurations for production concurrency
    engine = create_engine(
        settings.DATABASE_URL, 
        pool_size=20, 
        max_overflow=10, 
        pool_pre_ping=True
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database engine initialized successfully with pooling.")
except Exception as e:
    logger.critical(f"Critical Error: Could not initialize database engine: {e}")

# Database model for persistent logging of predictions
class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    input_data = Column(Text, nullable=False)
    prediction = Column(Integer)
    iso_score = Column(Float)
    mse_error = Column(Float)
    process_time_ms = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Ensure database tables are created at startup safely
@app.on_event("startup")
def on_startup():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema verified/created successfully at startup.")
    except Exception as e:
        logger.error(f"Error creating database schema at startup: {e}")

# Dependency to provide DB session per request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Load MLflow Hybrid Model from URI
logger.info(f"Attempting to load MLflow model from: {settings.MODEL_URI}")
try:
    model = mlflow.pyfunc.load_model(settings.MODEL_URI)
    logger.info("MLflow Hybrid Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Critical System Blocker: Model could not be fetched: {e}")

# Pydantic models for request validation and response formatting
class FraudInput(BaseModel):
    features: List[float] = Field(..., example=[0.1]*30)

class FraudResponse(BaseModel):
    id: str
    fraud_prediction: int
    iso_score: float
    reconstruction_error: float
    process_time_ms: float
    timestamp: datetime

# Key function for Rate Limiting: Identifies user via JWT 'sub' or IP address
def get_smart_identifier(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub")
            if user_id:
                return f"user:{user_id}"
        except Exception as e:
            logger.debug(f"JWT identifier extraction failed: {e}")
    return f"ip:{get_remote_address(request)}"

# Initialize Rate Limiter
limiter = Limiter(key_func=get_smart_identifier)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration for external integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS, 
    allow_credentials=True,                
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Middleware for monitoring latency via HTTP headers
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Standard Credit Card Dataset feature names (PCA components + Time/Amount)
FEATURE_NAMES = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

# --- API Endpoints ---

# FIXED: Changed from 'async def' to standard 'def' to offload Synchronous DB Blocking operations to ThreadPool
@app.post("/predict", response_model=FraudResponse)
@limiter.limit("10/minute")
def predict_fraud(data: FraudInput, request: Request, db: Session = Depends(get_db)):
    """
    Predicts if a transaction is fraudulent using the Hybrid (AE + ISO) model.
    Inputs exactly 30 features and safely offloads synchronous logging tasks.
    """
    start_time = time.time()
    logger.info("Incoming prediction request routing to inference framework...")

    try:
        # Validate feature matrix shape
        if len(data.features) != 30:
            logger.warning(f"Invalid input shape: Expected 30 features, got {len(data.features)}")
            raise ValueError("Exactly 30 features are required (PCA components + Time/Amount)")

        # Map list inputs to structural DataFrame matching the MLflow inferred schema
        input_df = pd.DataFrame([data.features], columns=FEATURE_NAMES)

        # Run model inference (The Custom PyFunc Wrapper handles Log Transform & Scaling automatically)
        raw_results = model.predict(input_df)

        pred = int(raw_results["fraud_prediction"].iloc[0])
        iso = float(raw_results["iso_score"].iloc[0])
        mse = float(raw_results["reconstruction_error"].iloc[0])

        process_time = (time.time() - start_time) * 1000
        request_id = str(uuid.uuid4())

        # Database Persistence Audit Logging
        new_log = PredictionLog(
            id=request_id,
            input_data=str(data.features),
            prediction=pred,
            iso_score=iso,
            mse_error=mse,
            process_time_ms=process_time
        )
        db.add(new_log)
        db.commit()

        logger.info(f"ID: {request_id} | Outcome: {'FRAUD' if pred == 1 else 'NORMAL'} | Latency: {process_time:.2f}ms")

        return {
            "id": request_id,
            "fraud_prediction": pred,
            "iso_score": iso,
            "reconstruction_error": mse,
            "process_time_ms": round(process_time, 2),
            "timestamp": datetime.now(timezone.utc)
        }

    except Exception as e:
        logger.error(f"Inference Application Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/logs/fraud-only")
def get_all_frauds(db: Session = Depends(get_db)):
    """Retrieves historical prediction matrices flagged strictly as fraud."""
    logger.info("Fetching verified fraud anomalies history records.")
    return db.query(PredictionLog).filter(PredictionLog.prediction == 1).all()

@app.put("/logs/{log_id}")
def update_prediction_status(log_id: str, new_pred: int, db: Session = Depends(get_db)):
    """Manual override mechanism for validation alignment (Human-in-the-loop strategy)."""
    log = db.query(PredictionLog).filter(PredictionLog.id == log_id).first()
    if not log:
        logger.warning(f"Override update execution failed: Log ID {log_id} missing.")
        raise HTTPException(status_code=404, detail="Log record not found")
    
    log.prediction = new_pred
    db.commit()
    logger.info(f"Human-in-the-loop override: Log ID {log_id} re-classified to: {new_pred}")
    return {"status": "Updated successfully"}

@app.delete("/logs/cleanup")
def delete_old_logs(db: Session = Depends(get_db)):
    """Clears all logged history elements from persistent database store."""
    logger.warning("Log cleanup protocol activated - purging all structural logs.")
    db.query(PredictionLog).delete()
    db.commit()
    return {"message": "All database logs cleared successfully"}

@app.get("/")
def health():
    """Liveness and Readiness infrastructure health probe indicator endpoint."""
    return {"status": "API is healthy and serving inference requests"}

if __name__ == "__main__":
    import uvicorn
    # Production execution engine configuration
    uvicorn.run(app, host="0.0.0.0", port=8000)
