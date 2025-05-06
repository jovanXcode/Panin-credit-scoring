from pydantic import BaseModel, ConfigDict, Field, UUID4
from datetime import datetime, date
from typing import Union, Any
from src.config.constant import DUMMY_DATASET_TABLE_NAME

class ModelBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID4 = Field(examples=["model-dummy-v0"])
    name: str = Field(examples=["model-dummy-v0"])
    run_id: str = Field(examples=["918495eaf2dd447497f368bb44b78c78"])
    status: str = Field(examples=["trained", "testing", "deployed"])
    dataset_table_name: str = Field(examples=[DUMMY_DATASET_TABLE_NAME])
    features: list[str] = Field(examples=[["job_title", "job_status", "income_type", "annual_income_class","income_class",'province','age']])
    algorithm_id: UUID4 = Field(examples=["1f92a42e-3509-40e1-b3e9-9e3c5f6e899e"])
    detail: dict[str, Any] = Field(examples=[{
        "risk_definition": {'dpd_m1'}, 
        "performance_metrics": {'accuracy' : 85}, 
        "threshold": 58, 
        "test_ratio": 0.1, 
        "test_n_month": 3,
        "train_start_date": "2025-02-01T11:12:00",
        "train_end_date": "2025-06-30T11:12:00",
        "test_start_date": "2025-07-01T11:12:00",
        "test_end_date": "2025-07-30T11:12:00"
    }])
    dataset_start_date: date = Field(examples=["2000-01-01"])
    dataset_end_date: date = Field(examples=["2050-12-31"])
    created_at: Union[None, datetime] = Field(examples=["2024-08-20T10:13:24.997944"])

class ModelResponse(BaseModel):

    id: UUID4 = Field(examples=["model-dummy-v0"])
    name: str = Field(examples=["model-dummy-v0"])
    run_id: str = Field(examples=["918495eaf2dd447497f368bb44b78c78"])
    status: str = Field(examples=["trained", "testing", "deployed"])
    dataset_table_name: str = Field(examples=[DUMMY_DATASET_TABLE_NAME])
    features: list[str] = Field(examples=[["job_title", "job_status", "income_type", "annual_income_class","income_class",'province','age']])
    algorithm_name: str = Field(examples=["LR-ScoreCard"])
    detail: dict[str, Any] = Field(examples=[{
        "risk_definition": {'dpd_m1'}, 
        "performance_metrics": {'accuracy' : 85}, 
        "threshold": 58, 
        "test_ratio": 0.1, 
        "test_n_month": 3,
        "train_start_date": "2025-02-01T11:12:00",
        "train_end_date": "2025-06-30T11:12:00",
        "test_start_date": "2025-07-01T11:12:00",
        "test_end_date": "2025-07-30T11:12:00"
    }])
    dataset_start_date: date = Field(examples=["2000-01-01"])
    dataset_end_date: date = Field(examples=["2050-12-31"])
    created_at: datetime = Field(examples=["2024-08-20T10:13:24.997944"])

    updated_at: Union[None, datetime] = Field(examples=["2024-08-20T10:13:24.997944"])
    deleted_at: Union[None, datetime] = Field(examples=["2024-08-20T10:13:24.997944"])

class ModelListResponse(BaseModel):
    models: list[ModelResponse] | None = None

class DeleteModelRequest(BaseModel):
    scoring_model_name: str = Field(examples=['model-alpha'])

class DeleteModelResponse(BaseModel):
    is_deleted: bool = Field(examples=["true"])

class AlgorithmBase(BaseModel):
    id: UUID4 = Field(examples=["model-dummy-v0"])
    name: str = Field(examples=["LR-ScoreCard"])
    description: str = Field(examples=["A model uses logistic regression to predict the likelihood of an event"])

class AlgorithmListResponse(BaseModel):
    algorithms: list[AlgorithmBase] = Field(examples=[
      [
        {
          "id": "1f92a42e-3509-40e1-b3e9-9e3c5f6e899e",
          "name": "LR-ScoreCard",
          "description": "A model uses logistic regression to predict the likelihood of an event; its advantage is that it is easy to understand and fast in processing data, while its drawback is that it cannot capture complex patterns and may be affected by highly correlated data."
        },
        {
          "id": "5yz2ae11-aac1-54a-ac5fg60a90da123cf11",
          "name": "Elasticnet-ScoreCard",
          "description": "A model combines two methods to make predictions more accurate and avoid errors; its advantage is that it is effective at selecting important features and works well with complicated data, while its drawback is that it requires careful tuning and can be hard to understand."
        },
        {
          "id": "19e2837e-9b79-430a-a3a5-e9f4b1ce26d4",
          "name": "LRCV-ScoreCard",
          "description": "A logistic regression model that uses a technique to test its accuracy; its advantage is that it gives more reliable results, but its drawback is that the testing process can take time and be complex."
        },
        {
          "id": "31d5b5e0-79ee-4b1c-8f0b-2b44e42f0034",
          "name": "ElasticnetCV-ScoreCard",
          "description": "A model uses Elastic Net with testing to find the best way to make predictions; its advantage is that it provides highly accurate results and automatically selects important features, but it can be hard to understand and requires deep knowledge."
        }
      ]
    ])
