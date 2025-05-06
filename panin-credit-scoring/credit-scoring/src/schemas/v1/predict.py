from datetime import datetime, date
from pydantic import BaseModel, ConfigDict, Field, UUID4
from typing import Union, Optional
from src.config.constant import DUMMY_DATASET_TABLE_NAME

class PredictRequest(BaseModel):
    scoring_model_name: str = Field(examples=["model_xai"])
    store_result: bool = Field(default=True, examples=['false'])
    app_id: str = Field(examples=['917ae282-15b1-470a-ac5f-60a90daecf30'])
    app_detail: dict = Field(examples=[{
        "customer_name": 'Bunga',
        "customer_national_id": 2984372128309824,
        "customer_gender": 'female',
        "customer_place_and_date_of_birth": 'Jakarta, 30 Januari 2000',
        "customer_address": 'Graha Lima Lima Jl. Tanah Abang II No.57 1, RT.1/RW.4',
        "customer_subdistrict" : 'Petojo Selatan',
        "customer_district" : 'Gambir',
        "customer_city" : 'Jakarta Pusat',
        "customer_postal_code" : 10160,
        "customer_email": 'wfoijqe@gmail.com',
        "customer_no_handphone": 6283248837191912
        }])
    input: list[dict] = Field(
        examples=[
            [
                {
                    "age": 22,
                    "marital_status": "MARRIED",
                    "occupation": "PROFESSIONAL"
                }
            ]
        ]
    )
    max_limit: Optional[float] = Field(default=None, examples=[100.0])
    min_limit: Optional[float] = Field(default=None, examples=[10.0])


class PredictResponse(BaseModel):
    predict_id: str = Field(examples=['917ae282-15b1-470a-ac5f-60a90daecf30'])
    app_id: str = Field(examples=['917ae282-15b1-470a-ac5f-60a90daecf30'])
    app_detail: dict = Field(examples=[{
        "customer_name": 'Bunga',
        "customer_national_id": 2984372128309824,
        "customer_gender": 'female',
        "customer_place_and_date_of_birth": 'Jakarta, 30 Januari 2000',
        "customer_address": 'Graha Lima Lima Jl. Tanah Abang II No.57 1, RT.1/RW.4',
        "customer_subdistrict" : 'Petojo Selatan',
        "customer_district" : 'Gambir',
        "customer_city" : 'Jakarta Pusat',
        "customer_postal_code" : 10160,
        "customer_email": 'wfoijqe@gmail.com',
        "customer_no_handphone": 6283248837191912
        }])
    psi: float = Field(examples=[0.12])
    kl_mean: float = Field(examples=[0.01])
    kl_median: float = Field(examples=[0.01])
    data_size: int = Field(examples=[923])
    result: list[dict] = Field(
        examples=[
            [
                {
                    "score": 36.2813046905,
                    "detail": {
                        "age": 5.9779441213,
                        "occupation": 11.5329293952,
                        "marital_status": 18.770431174
                    },
                    "recommended_limit" : 9390000,
                    "recommended_decision" :"ACCEPT"
                }
            ]
        ]
    )
    result_status: str = Field(examples=['success', 'failed'])
    predicted_at: datetime = Field(examples=['2024-08-20T10:13:24.997944'])
    min_risk_score: int = Field(examples=[0])
    max_risk_score: int = Field(examples=[100])
    range: list[int] = Field(examples=[[0, 25, 50, 75, 100]])

class PredictHistoryBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    predict_id: UUID4 = Field(examples=['917ae282-15b1-470a-ac5f-60a90daecf30'])
    scoring_model_name: str = Field(examples=['model-xai'])
    app_id: str = Field(examples=['917ae282-15b1-470a-ac5f-60a90daecf30'])
    app_detail: dict = Field(examples=[{
        "customer_name": 'Bunga',
        "customer_national_id": 2984372128309824,
        "customer_gender": 'female',
        "customer_place_and_date_of_birth": 'Jakarta, 30 Januari 2000',
        "customer_address": 'Graha Lima Lima Jl. Tanah Abang II No.57 1, RT.1/RW.4',
        "customer_subdistrict" : 'Petojo Selatan',
        "customer_district" : 'Gambir',
        "customer_city" : 'Jakarta Pusat',
        "customer_postal_code" : 10160,
        "customer_email": 'wfoijqe@gmail.com',
        "customer_no_handphone": 6283248837191912
        }])
    output_table_name: str = Field(examples=['scoring_result'])
    input: list[dict] = Field(
        examples=[
            [
                {
                    "age": 22,
                    "marital_status": "married"
                }
            ]
        ]
    )
    result: list[dict] = Field(
        examples=[
            [
                {
                    "score": 36.2813046905,
                    "detail": {
                        "age": 5.9779441213,
                        "occupation": 11.5329293952,
                        "marital_status": 18.770431174
                    },
                    "recommended_limit" :9390000,
                    "recommended_decision" :"ACCEPT"
                }
            ]
        ]
    )
    detail: dict = Field(examples=[{'min_score': 10, 'max_score':72}])
    predicted_at: datetime = Field(examples=['2024-08-20T10:13:24.997944'])

class PredictHistoryListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    result: list[PredictHistoryBase] | None = None

class MonitorHistoryBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    monitor_id: UUID4 = Field(examples=['5yz2ae11-aac1-54a-ac5fg60a90da123cf11'])
    monitored_at: datetime = Field(examples=['2024-08-20T10:13:24.997944'])
    scoring_model_name: str = Field(examples=['model-xai'])
    psi: float = Field(examples=[0.11])
    kl_mean: float = Field(examples=[0.01])
    kl_median: float = Field(examples=[0.01])
    input_drift_detail: dict = Field(examples=[{
        "age": {
            "kl": 0.01,
            "is_significant": False,
            "type": "Numerical"
        },
        "debt_to_income": {
            "kl": 0.02,
            "is_significant": False,
            "type": "Numerical"
        },
        "education": {
            "kl": 0.001,
            "is_significant": False,
            "type": "Categorical"
        },
        "annual_income_class": {
            "kl": 0.005,
            "is_significant": False,
            "type": "Numerical"
        },
        "income_class": {
            "kl": 0.015,
            "is_significant": False,
            "type": "Numerical"
        },
        "income_type": {
            "kl": 0.008,
            "is_significant": False,
            "type": "Categorical"
        },
        "interest_rate": {
            "kl": 0.04,
            "is_significant": False,
            "type": "Numerical"
        },
        "job_status": {
            "kl": 0.02,
            "is_significant": False,
            "type": "Categorical"
        },
        "job_title": {
            "kl": 0.006,
            "is_significant": False,
            "type": "Categorical"
        },
        "length_of_stay": {
            "kl": 0.02,
            "is_significant": False,
            "type": "Numerical"
        },
        "remaining_amount_loan": {
            "kl": 0.005,
            "is_significant": False,
            "type": "Numerical"
        },
        "marital_status": {
            "kl": 0.004,
            "is_significant": False,
            "type": "Categorical"
        },
        "num_active_loans": {
            "kl": 0.01,
            "is_significant": False,
            "type": "Numerical"
        },
        "num_dependent": {
            "kl": 0.007,
            "is_significant": False,
            "type": "Numerical"
        },
        "num_work_year": {
            "kl": 0.015,
            "is_significant": False,
            "type": "Numerical"
        },
        "occupation": {
            "kl": 0.009,
            "is_significant": False,
            "type": "Categorical"
        },
        "province": {
            "kl": 0.012,
            "is_significant": False,
            "type": "Categorical"
        },
        "latest_payment_amount": {
            "kl": 0.003,
            "is_significant": False,
            "type": "Numerical"
        }
        }])
    num_account: int = Field(examples=[123547])
    min_score: float = Field(examples=[1])
    max_score: float = Field(examples=[100])
    avg_score: float = Field(examples=[45])
    median_score: float = Field(examples=[50])
    std_score: float = Field(examples=[14])

class MonitorHistoryListBase(BaseModel):
    history: list[MonitorHistoryBase] | None = None

class AggregatedMonitorBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    scoring_model_name: str = Field(examples=['model-xai'])
    date: str = Field(examples=['2024-03'])
    
    psi: float = Field(examples=[0.11])
    kl_mean: float = Field(examples=[0.01])
    kl_median: float = Field(examples=[0.01])
    min_score: float = Field(examples=[1])
    max_score: float = Field(examples=[100])
    avg_score: float = Field(examples=[45])
    median_score: float = Field(examples=[50])
    std_score: float = Field(examples=[14])

class AggregatedMonitorHistoryBase(BaseModel):
    history: list[AggregatedMonitorBase] | None = None

class PredictScheduleBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    scoring_model_name: str = Field(examples=['model-xai'])
    input_table_name: str = Field(examples=['sample_dataset'])
    input_start_date: date | None = Field(default='1900-01-01', examples=["1900-01-01"])
    input_end_date: date | None = Field(default='2100-01-01', examples=["2500-12-31"])
    cron_format: str = Field(examples=['* * * * *'])

class PredictScheduleRequest(PredictScheduleBase):
    model_config = ConfigDict(from_attributes=True)

    schedule_id: UUID4 = Field(examples=['917ae282-15b1-470a-ac5f-60a90daecf30'])
    taks_key: str = Field(examples=['82-1560a90daecf30b1-470a'])
    created_at: datetime = Field(examples=['2024-08-20T10:13:24.997944'])

class PredictScheduleResponse(PredictScheduleRequest):
    model_config = ConfigDict(from_attributes=True)

    updated_at: Union[None, datetime] = Field(examples=["2024-08-20T10:13:24.997944"])
    deleted_at: Union[None, datetime] = Field(examples=["2024-08-20T10:13:24.997944"])

class PredictScheduleListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    result: list[PredictScheduleResponse] | None = None

class DeleteScheduleRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    scoring_model_name: str = Field(examples=["model_xai"])

class DeleteScheduleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    is_deleted: bool = Field(examples=[True])