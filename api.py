import logging
import time
import uuid
from fastapi import Request
import logging
import pandas as pd
import mlflow.pyfunc
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi.middleware.cors import CORSMiddleware
import jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
import logging
from config import settings


# FastAPI app setup
app = FastAPI(title="Enterprise Fraud Detection API") # لازم السطر ده يكون موجود كدة بالظبط

# logging file setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("fraud_api.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AnomalyDetectionAPI")

# Database setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model for logging predictions
class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    input_data = Column(Text, nullable=False)
    prediction = Column(Integer)
    iso_score = Column(Float)
    mse_error = Column(Float)
    process_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Load MLflow model
try:
    model = mlflow.pyfunc.load_model(settings.MODEL_URI)
    logger.info("✅ MLflow Hybrid Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load model: {str(e)}")
    raise

# Pydantic models for request and response
class FraudInput(BaseModel):
    features: List[float] = Field(..., example=[0.1]*30)


class FraudResponse(BaseModel):
    id: str
    fraud_prediction: int
    iso_score: float
    reconstruction_error: float
    process_time_ms: float
    timestamp: datetime


# Rate limiting key function that checks for user ID in JWT or falls back to IP address
def get_smart_identifier(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub")
            if user_id:
                return f"user:{user_id}"
        except:
            pass
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=get_smart_identifier)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Middle & CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS, 
    allow_credentials=True,                
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


FEATURE_NAMES = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]


# API Endpoints
@app.post("/predict")
@limiter.limit("10/minute")
async def predict_fraud(data: FraudInput, request: Request, db: Session = Depends(get_db)):
    start_time = time.time()

    try:
        if len(data.features) != 30:
            raise ValueError("Exactly 30 features are required (PCA components + Time/Amount)")

        # التعديل هنا: إضافة columns=FEATURE_NAMES
        input_df = pd.DataFrame([data.features], columns=FEATURE_NAMES)

        # دلوقتى MLflow هيعرف يطابق البيانات مع الـ Schema بتاعته
        raw_results = model.predict(input_df)

        pred = int(raw_results["fraud_prediction"].iloc[0])
        iso = float(raw_results["iso_score"].iloc[0])
        mse = float(raw_results["reconstruction_error"].iloc[0])

        process_time = (time.time() - start_time) * 1000
        request_id = str(uuid.uuid4())

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

        logger.info(f"ID: {request_id} | Pred: {pred} | Time: {process_time:.2f}ms")

        return {
            "id": request_id,
            "fraud_prediction": pred,
            "iso_score": iso,
            "reconstruction_error": mse,
            "process_time_ms": round(process_time, 2),
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Additional endpoints for logs management
@app.get("/logs/fraud-only")
def get_all_frauds(db: Session = Depends(get_db)):
    return db.query(PredictionLog).filter(PredictionLog.prediction == 1).all()


@app.put("/logs/{log_id}")
def update_prediction_status(log_id: str, new_pred: int, db: Session = Depends(get_db)):
    log = db.query(PredictionLog).filter(PredictionLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    log.prediction = new_pred
    db.commit()
    return {"status": "Updated successfully"}


@app.delete("/logs/cleanup")
def delete_old_logs(db: Session = Depends(get_db)):
    db.query(PredictionLog).delete()
    db.commit()
    return {"message": "All logs cleared"}

@app.get("/")
def health():
    return {"status": "API is running 🚀"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)