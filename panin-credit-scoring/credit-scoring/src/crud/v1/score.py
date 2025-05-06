import logging
import datetime
from sqlalchemy import desc, extract, func
from sqlalchemy.orm import Session

from src.models.v1.scoring_history import ScoringHistory, MonitorHistory
from src.schemas.v1.predict import PredictHistoryBase, MonitorHistoryBase

logger = logging.getLogger(__name__)

def add_scoring_history(db: Session, predict_history_record: PredictHistoryBase):
    new_record = ScoringHistory(**predict_history_record.model_dump())

    try:
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
    except Exception as e:
        logger.exception('db: add scoring history error')
        db.rollback()
        raise e

    return new_record

def get_scoring_history(db: Session):
    return db.query(ScoringHistory).order_by(desc(ScoringHistory.predicted_at)).all()

def add_monitoring_history(db: Session, monitor_history_record: MonitorHistoryBase):
    new_record = MonitorHistory(**monitor_history_record.model_dump())

    try:
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
    except Exception as e:
        logger.exception('db: add monitoring history error')
        db.rollback()
        raise e

    return new_record

def upsert_monitoring_history(db: Session, monitor_history_record: MonitorHistoryBase):
    try:
        existing_record = db.query(MonitorHistory).filter(
            extract('day', MonitorHistory.monitored_at) == 25,
            extract('month', MonitorHistory.monitored_at) == datetime.datetime.today().month,
            extract('year', MonitorHistory.monitored_at) == datetime.datetime.today().year  
        ).first()

        if existing_record:
            # Update the dataset attributes
            existing_record.psi = monitor_history_record.psi
            existing_record.num_account = monitor_history_record.num_account
            existing_record.kl_mean = monitor_history_record.kl_mean
            existing_record.kl_median = monitor_history_record.kl_median
            existing_record.input_drift_detail = monitor_history_record.input_drift_detail
            existing_record.num_account = monitor_history_record.num_account
            existing_record.min_score = monitor_history_record.min_score
            existing_record.max_score = monitor_history_record.max_score
            existing_record.avg_score = monitor_history_record.avg_score
            existing_record.median_score = monitor_history_record.median_score
            existing_record.std_score = monitor_history_record.std_score

            db.commit()
            return existing_record
        else:
            return add_monitoring_history(db, monitor_history_record)
    
    except Exception as e:
        logger.exception('db: upsert monitoring history error')
        db.rollback()
        raise e

def get_monitoring_history(db: Session, model_name: str):
    return db.query(MonitorHistory).filter(MonitorHistory.scoring_model_name == model_name).order_by(desc(MonitorHistory.monitored_at)).all()

def get_latest_monitoring_history(db: Session, model_name: str, n_months: int):
    result = db.query(
        MonitorHistory.scoring_model_name,
        func.to_char(MonitorHistory.monitored_at, 'YYYY-MM').label('date'),
        func.min(MonitorHistory.psi).label('psi'),
        func.min(MonitorHistory.kl_mean).label('kl_mean'),
        func.min(MonitorHistory.kl_median).label('kl_median'),
        func.min(MonitorHistory.min_score).label('min_score'),
        func.min(MonitorHistory.max_score).label('max_score'),
        func.min(MonitorHistory.avg_score).label('avg_score'),
        func.min(MonitorHistory.median_score).label('median_score'),
        func.min(MonitorHistory.std_score).label('std_score')
    ).filter(
        MonitorHistory.scoring_model_name == model_name,
        MonitorHistory.monitored_at > (datetime.datetime.today().replace(day=1, hour=0, minute=0) - datetime.timedelta(days=30 * n_months))
    ).group_by(
        func.to_char(MonitorHistory.monitored_at, 'YYYY-MM'),
        MonitorHistory.scoring_model_name
    ).order_by(
        desc(func.to_char(MonitorHistory.monitored_at, 'YYYY-MM'))
    ).all()

    return result

def get_latest_monitornig_date(db: Session, model_name: str):
    result = db.query(
        func.max(MonitorHistory.monitored_at)
    ).filter(
        MonitorHistory.scoring_model_name == model_name
    ).first()

    return result