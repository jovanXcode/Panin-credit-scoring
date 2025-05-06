import logging
import uuid
from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from src.models.v1.model import Model, ModelTrainingStatus, Algorithm
from src.schemas.v1.model import ModelBase
from src.schemas.v1.train import TrainStatusBase

logger = logging.getLogger(__name__)

# def create_model(db: Session, model: ModelBase):
#     new_model = Model(**model.model_dump())
#     try:
#         db.add(new_model)
#         db.commit()
#         db.refresh(new_model)
#     except Exception as e:
#         logger.exception('db: create new model error')
#         db.rollback()
#         raise e

#     return new_model

def create_model(db: Session, model: ModelBase):
    new_model = Model(**model.model_dump())

    # Create an upsert statement
    stmt = insert(Model).values(**model.model_dump()).on_conflict_do_update(
        index_elements=['name'],  # Specify the unique constraint column(s)
        set_={
            "run_id": new_model.run_id,
            "status": new_model.status,
            "features": new_model.features,
            "algorithm_id": new_model.algorithm_id,
            "detail": new_model.detail,
            "dataset_table_name": new_model.dataset_table_name,
            "dataset_start_date": new_model.dataset_start_date,
            "dataset_end_date": new_model.dataset_end_date,
            "created_at": new_model.created_at,
            "updated_at": new_model.updated_at,
            "deleted_at": new_model.deleted_at,
        }
    )

    try:
        db.execute(stmt)
        db.commit()
        #db.refresh(new_model)
    except Exception as e:
        logger.exception('db: create or update model error')
        db.rollback()
        raise e

    return new_model

def delete_model(db: Session, model_name: str):
    try:
        db.query(Model).filter(Model.name == model_name, Model.deleted_at.is_(None)).delete()
        db.commit()
    except Exception as e:
        logger.exception('db: delete model error')
        db.rollback()
        raise e

    return True

def soft_delete_model(db: Session, model_name: str):
    try:
        db.query(Model).filter(Model.name == model_name).update({Model.deleted_at: datetime.now()})
        db.commit()
    except Exception as e:
        logger.exception('db: soft delete model error')
        db.rollback()
        raise e

    return True 

def get_model_by_name(db: Session, model_name: str):
    return db.query(Model).filter(Model.name == model_name, Model.deleted_at.is_(None)).one()

def get_models(db: Session):
    return db.query(Model).filter(Model.deleted_at.is_(None)).all()

def get_models_by_status(db: Session, status: str):
    return db.query(Model).filter(Model.status == status, Model.deleted_at.is_(None)).all()

def update_model_status(db: Session, model_name: str, status: str):
    try:
        db.query(Model).filter(Model.name == model_name).update({Model.status: status})
        db.commit()
    except Exception as e:
        logger.exception('db: update model status error')
        db.rollback()
        raise e
    
    return True

def create_train_status(db: Session, record: TrainStatusBase):
    new_task = ModelTrainingStatus(**record.model_dump())
    try:
        db.add(new_task)
        db.commit()
        db.refresh(new_task)
    except Exception as e:
        logger.exception('db: create new train status error')
        db.rollback()

    return new_task

def get_train_status(db: Session):
    return db.query(ModelTrainingStatus).filter(ModelTrainingStatus.deleted_at.is_(None)).all()

def get_train_status_by_model_name(db: Session, scoring_model_name: str):
    return db.query(ModelTrainingStatus).filter(ModelTrainingStatus.scoring_model_name == scoring_model_name, 
                                                ModelTrainingStatus.deleted_at.is_(None)).first()

def get_train_status_by_task_id(db: Session, task_id: str):
    return db.query(ModelTrainingStatus).filter(ModelTrainingStatus.task_id == task_id, 
                                                ModelTrainingStatus.deleted_at.is_(None)).first()

def update_train_status(db: Session, task_id: str, task_status: str, status_code: int, status_code_detail: str):
    try:
        record = {
            ModelTrainingStatus.task_status: task_status,
            ModelTrainingStatus.status_code: status_code,
            ModelTrainingStatus.status_code_detail: status_code_detail
        }
        db.query(ModelTrainingStatus).filter(ModelTrainingStatus.task_id == task_id).update(record)
        db.commit()
    except Exception as e:
        logger.exception('db: update train status error')
        db.rollback()
        raise e
    
    return get_train_status_by_task_id(db, task_id)

def get_algorithms(db: Session):
    return db.query(Algorithm).all()

def get_algorithm_by_id(db: Session, algorithm_id: uuid.UUID) -> str | None:
    result = db.query(Algorithm).filter(Algorithm.id == algorithm_id).first()
    return result

def update_threshold_by_model(db: Session, model_name: str, threshold: float):
    try:
        db.query(Model).filter(Model.name == model_name).update({
            Model.detail: func.jsonb_set(Model.detail, '{threshold}', func.to_jsonb(threshold))
        })
        db.commit()
    except Exception as e:
        logger.exception('db: soft delete model error')
        db.rollback()
        raise e

    return True 