from datetime import datetime
from sqlalchemy import Column, func
from sqlalchemy import Integer, String, Uuid, DateTime, Date, Float
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from src.utils.database import Base

class ScoringHistory(Base):
    __tablename__ =  "scoring_history"

    predict_id = Column(Uuid, primary_key=True)
    app_id = Column(String)
    scoring_model_name = Column(String)
    app_detail= Column(JSONB)
    output_table_name = Column(String)
    input = Column(ARRAY(JSONB))
    result = Column(ARRAY(JSONB))
    detail = Column(JSONB)
    predicted_at = Column(DateTime)

class MonitorHistory(Base):
    __tablename__ = "monitor_history"

    monitor_id = Column(Uuid, primary_key=True)
    monitored_at = Column(DateTime, default=datetime.now)
    scoring_model_name = Column(String)

    psi = Column(Float)
    kl_mean = Column(Float)
    kl_median = Column(Float)
    input_drift_detail = Column(JSONB)
    num_account = Column(Integer)
    min_score = Column(Float)
    max_score = Column(Float)
    avg_score = Column(Float)
    median_score = Column(Float)
    std_score = Column(Float)
    