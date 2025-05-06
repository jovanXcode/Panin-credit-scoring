import logging
from datetime import datetime
from sqlalchemy.orm import Session

from src.models.v1.dataset import Dataset
from src.schemas.v1.dataset import CreateDatasetBase, UpdateDatasetRequest

logger = logging.getLogger(__name__)

def create_dataset(db: Session, dataset: CreateDatasetBase):
    new_dataset = Dataset(**dataset.model_dump())
    try:
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)
    except Exception as e:
        logger.exception('db: create dataset error')
        db.rollback()
        raise e

    return new_dataset

def delete_dataset(db: Session, table_name: str):
    try:
        db.query(Dataset).filter(Dataset.table_name == table_name).delete()
        db.commit()
    except Exception as e:
        logger.exception('db: delete dataset error')
        db.rollback()
        raise e

    return True

def update_dataset(db: Session, req: UpdateDatasetRequest):
    try:
        existing_dataset = db.query(Dataset).filter(Dataset.table_name == req.table_name).first()

        if existing_dataset:
            # Update the dataset attributes
            existing_dataset.target_column_names = req.target_column_names
            existing_dataset.identity_column_names = req.identity_column_names
            existing_dataset.datetime_column_names = req.datetime_column_names
            existing_dataset.description = req.description
            existing_dataset.updated_at = datetime.now()

            db.commit()
            return existing_dataset
        else:
            return None
    except Exception as e:
        logger.exception('db: update dataset error')
        db.rollback()
        raise e

def get_dataset_by_name(db: Session, table_name: str):
    return db.query(Dataset).filter(Dataset.table_name == table_name, Dataset.deleted_at.is_(None)).first()

def get_datasets(db: Session):
    return db.query(Dataset).filter(Dataset.deleted_at.is_(None)).all()
