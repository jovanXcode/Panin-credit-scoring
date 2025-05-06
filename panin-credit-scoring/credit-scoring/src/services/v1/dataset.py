import logging
import psycopg2
import uuid
from datetime import datetime
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from typing import Union
import pandas as pd

from src.schemas.v1.dataset import CreateDatasetRequest, UpdateDatasetRequest
from src.schemas.v1.dataset import CreateDatasetBase
from src.crud.v1 import dataset as repo_dataset

from src.config.constant import DB_POSTGRES_USER 
from src.config.constant import DB_POSTGRES_PASSWORD
from src.config.constant import DB_POSTGRES_HOST
from src.config.constant import DB_POSTGRES_PORT
from src.config.constant import DB_POSTGRES_NAME

logger = logging.getLogger(__name__)

def get_all_datasets(db: Session):
    return repo_dataset.get_datasets(db)

def get_dataset_by_name(db: Session, table_name: str):
    return repo_dataset.get_dataset_by_name(db, table_name)

def create_dataset(db: Session, req: CreateDatasetRequest):
    validate_table(db, req)

    new_dataset = CreateDatasetBase(
        id = str(uuid.uuid4()),
        table_name = req.table_name,
        target_column_names=req.target_column_names,
        identity_column_names=req.identity_column_names,
        description=req.description,
        created_at = datetime.now()
    )
    return repo_dataset.create_dataset(db, new_dataset)

def delete_dataset(db: Session, table_name: str):
    return repo_dataset.delete_dataset(db, table_name)

def update_dataset(db: Session, req: UpdateDatasetRequest):
    validate_table(db, req)

    return repo_dataset.update_dataset(db, req)

def check_table(db: Session, table_name: str):
    logger.info(f'check [{table_name}] if exist')
    
    data_type_mapping = {
        'character varying': 'string',
        'double precision': 'float'
    }

    error_ = None
    is_exist = True 
    n_cols = 0
    table_schema = dict()
    try:
        conn = psycopg2.connect(user=DB_POSTGRES_USER, password=DB_POSTGRES_PASSWORD, 
                                host=DB_POSTGRES_HOST, port=DB_POSTGRES_PORT,
                                database=DB_POSTGRES_NAME)

        cursor = conn.cursor()

        cursor.execute(f'SELECT * FROM {table_name} LIMIT 1')
        _ = cursor.fetchone()    
    
        cursor.execute(
            f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"
        )
        table_info = cursor.fetchall()
        for col_name, col_type in table_info:
            table_schema[col_name] = data_type_mapping.get(col_type, col_type)
        
        n_cols = len(table_info)
    except (Exception, psycopg2.Error) as e:
        logger.exception('Check DB Error')
        is_exist = False 
        error_ = e
    finally:
        if conn:
            logger.info('closing db connection')
            cursor.close()
            conn.close()

        if error_:
            logger.warning(f"Table [{table_name}] does not exist or failed to fetch schema.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Table [{table_name}] not found."
            )

    result = {
        'is_exist': is_exist,
        'n_cols': n_cols, 
        'table_schema': table_schema
    }
    return result

def validate_table(db: Session, req: Union[CreateDatasetRequest, UpdateDatasetRequest]):
    check_result = check_table(db, req.table_name)
    if not check_result['is_exist']:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='table does not exist')

    for col in req.target_column_names:
        if col not in check_result['table_schema'].keys():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                                detail=f'invalid target column [{col}]')
    
    for col in req.identity_column_names:
        if col not in check_result['table_schema'].keys():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                                detail=f'invalid identity column [{col}]')
        
    for col in req.datetime_column_names:
        if col not in check_result['table_schema'].keys():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                                detail=f'invalid datetime column [{col}]')
        
        col_data_type = check_result['table_schema'][col]
        if 'time' not in col_data_type:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                                detail=f'invalid datetime column [{col}], given: {col_data_type}, expected: datetime/timestamp')
        


def get_dataset_df(db: Session, table_name: str, column_name:str):
    error_ = None
    try:
        conn = psycopg2.connect(user=DB_POSTGRES_USER, password=DB_POSTGRES_PASSWORD, 
                                host=DB_POSTGRES_HOST, port=DB_POSTGRES_PORT,
                                database=DB_POSTGRES_NAME)

        cursor = conn.cursor()

        cursor.execute(f'SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL;')
        # Fetch all data
        data = cursor.fetchall()

        features = [row[0] for row in data]

    
    except (Exception, psycopg2.Error) as e:
        logger.exception('Check DB Error')
      
        error_ = e
    finally:
        if conn:
            logger.info('closing db connection')
            cursor.close()
            conn.close()

        if error_: 
            raise error_
        
    return features


