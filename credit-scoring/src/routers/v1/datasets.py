import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.schemas.v1.dataset import DatasetSchemaResponse, DeleteDatasetResponse
from src.schemas.v1.dataset import CreateDatasetRequest, DeleteDatasetRequest
from src.schemas.v1.dataset import DatasetResponse, DatasetListResponse
from src.schemas.v1.dataset import UpdateDatasetRequest

from src.services.v1 import dataset as dataset_svc

from src.utils.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/v1/datasets',
    tags=['datasets'],
    responses={
        status.HTTP_400_BAD_REQUEST: {"message": "Bad request"}, 
        status.HTTP_404_NOT_FOUND: {"message": "Not found"}, 
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"message": "Unprocessable entity"}, 
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"message": "Server error"}
    },
)

@router.get("/", response_model=DatasetListResponse,
            summary='get all datasets')
async def get_datasets(db: Session = Depends(get_db)):
    datasets = dataset_svc.get_all_datasets(db)
    return {
        'datasets': datasets
    }

@router.get("/{dataset_name}", response_model=DatasetResponse,
            summary='get dataset detail')
async def get_dataset_detail(dataset_name: str, db: Session = Depends(get_db)):
    result = dataset_svc.get_dataset_by_name(db, dataset_name)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"Dataset [{dataset_name}] does not exist")
    return result

@router.post("/", status_code=status.HTTP_201_CREATED, response_model=DatasetResponse,
             summary='add or register a new dataset')
async def add_dataset(req: CreateDatasetRequest, db: Session = Depends(get_db)):
    old_dataset = dataset_svc.get_dataset_by_name(db, req.table_name)
    if old_dataset:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                            detail=f"Dataset [{req.table_name}] is already added")

    check_table = dataset_svc.check_table(db, req.table_name)
    if not check_table['is_exist']:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"Dataset [{req.table_name}] does not exist")

    return dataset_svc.create_dataset(db, req)

@router.delete("/", response_model=DeleteDatasetResponse,
               summary='delete a dataset')
async def delete_dataset(req: DeleteDatasetRequest, db: Session = Depends(get_db)):
    old_dataset = dataset_svc.get_dataset_by_name(db, req.table_name)
    if not old_dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"Dataset [{req.table_name}] does not exist")

    is_deleted = dataset_svc.delete_dataset(db, req.table_name)
    return DeleteDatasetResponse(is_deleted = is_deleted)

@router.put("/", response_model=DatasetResponse,
               summary='update a dataset')
async def update_dataset(req: UpdateDatasetRequest, db: Session = Depends(get_db)):
    old_dataset = dataset_svc.get_dataset_by_name(db, req.table_name)
    if not old_dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"Dataset [{req.table_name}] does not exist")

    result = dataset_svc.update_dataset(db, req)
    return result

@router.get("/schema/{table_name}", response_model=DatasetSchemaResponse,
            summary='check if the table exists and provide table schema information (column name: data type)')
async def check_table_schema(table_name: str, db: Session = Depends(get_db)):
    return dataset_svc.check_table(db, table_name)
