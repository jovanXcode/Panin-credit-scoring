from pydantic import BaseModel, ConfigDict, Field, UUID4
from typing import Dict, Union
from datetime import datetime
from src.config.constant import DUMMY_DATASET_TABLE_NAME

class DatasetSchemaResponse(BaseModel):
    is_exist: bool = Field(examples=["true"])
    n_cols: int = Field(examples=[12])
    # Conflicting namespace below, change schema
    table_schema: Dict[str, str] = Field(examples=[{'age': 'integer', 'province': 'gorontalo'}])

class DatasetBase(BaseModel):
    table_name: str = Field(examples=[DUMMY_DATASET_TABLE_NAME], default=DUMMY_DATASET_TABLE_NAME)
    target_column_names: list[str] = Field(default=["dpd_m1","col_m1"])
    identity_column_names: list[str] = Field(default=["customer_id", "loan_id"])
    datetime_column_names: list[str] = Field(default=["datetime"])
    description: str = Field(examples=['Training dataset'])

class CreateDatasetBase(DatasetBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID4 = Field(examples=["model-dummy-v0"])
    created_at: datetime = Field(examples=["2024-08-20T10:13:24.997944"])

class UpdateDatasetRequest(BaseModel):
    table_name: str = Field(examples=[DUMMY_DATASET_TABLE_NAME])
    target_column_names: list[str] = Field(default=["dpd_m1","col_m1"])
    identity_column_names: list[str] = Field(default=["customer_id", "loan_id"])
    datetime_column_names: list[str] = Field(default=["datetime"])
    description: str = Field(examples=['Training dataset'])

class DatasetResponse(CreateDatasetBase):
    model_config = ConfigDict(from_attributes=True)

    updated_at: Union[None, datetime] = Field(examples=["2024-08-20T10:13:24.997944"])
    deleted_at: Union[None, datetime] = Field(examples=["2024-08-20T10:13:24.997944"])

class DatasetListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    datasets: list[DatasetResponse] | None = None

class CreateDatasetRequest(DatasetBase):
    pass

class DeleteDatasetRequest(BaseModel):
    table_name: str = Field(examples=[DUMMY_DATASET_TABLE_NAME])

class DeleteDatasetResponse(BaseModel):
    is_deleted: bool = Field(examples=["true"])