import logging
import pandas as pd
from fastapi import status, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound

import src.crud.v1.model as model_crud
from src.schemas.v1.model import ModelBase

logger = logging.getLogger(__name__)

def get_models(db: Session):
    return model_crud.get_models(db)

def get_model_by_name(db: Session, model_name: str):
    return model_crud.get_model_by_name(db, model_name) 
        
def delete_model(db: Session, model_name: str):
    try:
        _ = get_model_by_name(db, model_name)
    except NoResultFound:
        msg = f'[{model_name}] not found'
        logger.info(msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)

    try:
        logger.info(f'deleting model [{model_name}] from db')
        is_deleted = model_crud.delete_model(db, model_name)
    except Exception as e:
        logger.exception(f'delete model [{model_name}] from db error')
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail='delete model error')

    return is_deleted

def upsert_model(db:Session, req: ModelBase):
    _ = model_crud.create_model(db, req)
    return {"is_success": True}

def get_algorithms(db:Session):
    return model_crud.get_algorithms(db)