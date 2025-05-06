from sqlalchemy import Column, func
from sqlalchemy import String, DateTime, Date, Uuid
from src.utils.database import Base

class ScoringSchedule(Base):
    __tablename__ =  "scoring_schedule"

    schedule_id = Column(Uuid, primary_key=True)
    scoring_model_name = Column(String)
    input_table_name = Column(String)
    input_start_date = Column(Date)
    input_end_date = Column(Date)
    taks_key = Column(String)
    cron_format = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    deleted_at = Column(DateTime)