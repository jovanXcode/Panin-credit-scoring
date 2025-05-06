from pydantic import BaseModel, Field, ConfigDict, UUID4, field_validator
from typing import Union, Any, Optional
from datetime import date, datetime

from src.config.constant import DUMMY_DATASET_TABLE_NAME

class UpdateFailedTrainRequest(BaseModel):
    scoring_model_name: str = Field(examples=['model-test-1'])
    response_status_code: int = Field(examples=[500])
    response_detail: dict

class TrainRequest(BaseModel):
    scoring_model_name: str = Field(examples=['model-test-1'])
    table_name: str = Field(examples=[DUMMY_DATASET_TABLE_NAME])
    start_date: date | None = Field(examples=["1900-01-01"])
    end_date: date | None= Field(examples=["2500-12-31"])
    features: list[str] = Field(examples=[['occupation', 'age', 'marital_status']])
    algorithm_id: UUID4 = Field(examples=["1f92a42e-3509-40e1-b3e9-9e3c5f6e899e"])
    selected_risk_definition: str = Field(examples=['Custom'])
    risk_definition: Union[dict, str] = Field(examples=[{
                "and": {
                    "dpd_m1": {"condition": ">", "value": 1},
                    "col_m1": {"condition": ">", "value": 2}
                }   
            }])
    test_ratio: Optional[float] = Field(default=0.2, examples=[0.2])
    test_n_month: Optional[int] = Field(default=-1, examples=[-1])
   
    @field_validator("risk_definition")
    def ensure_dict_or_str(cls, v):
        if isinstance(v, dict):
            return v  # accept dict
        return str(v)  # If the value is not dict, cast to string
    
    @field_validator("test_ratio")
    def validate_test_ratio(cls, v):
        if v in [0, -1]:
            return v
        if v < 0.05:
            raise ValueError("test_ratio must be at least 0.05 (5% of data)")
        if v > 0.5:
            raise ValueError("test_ratio must not exceed 0.5 (50% of data)")
        return v
    
    @field_validator("test_n_month")
    def validate_test_n_month(cls, v):
        if v in [0, -1]:
            return v
        # Apply constraints for non-default values
        if v < 1:
            raise ValueError("test_n_month must be at least 1 month")
        if v > 12:
            raise ValueError("test_n_month must not exceed 12 months")
        return v

class TrainStatusBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: int = Field(examples=[318])
    task_status: str = Field(examples=['PENDING'])
    scoring_model_name: str = Field(examples=['model-abc'])
    created_at: Union[None, datetime] = Field(examples=["2024-08-20T10:13:24.997944"])

class TrainStatusResponse(TrainStatusBase):
    model_config = ConfigDict(from_attributes=True)
    
    updated_at: Union[None, datetime] = Field(examples=["2024-08-20T10:13:24.997944"])
    deleted_at: Union[None, datetime] = Field(examples=["2024-08-20T10:13:24.997944"])

class TrainStatusListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    result: list[TrainStatusResponse] | None = None

class SimulationRequest(BaseModel):
    scoring_model_name: str = Field(examples=['model-test-1'])
    threshold_train: Optional[float] = Field(default=None, examples=[50])
    weight_fp_train: Optional[float] = Field(default=None, examples=[0])
    weight_fn_train: Optional[float] = Field(default=None, examples=[-1])
    weight_tn_train: Optional[float] = Field(default=None, examples=[0.35])
    weight_tp_train: Optional[float] = Field(default=None, examples=[0])
    threshold_test: Optional[float] = Field(default=None, examples=[50])
    weight_fp_test: Optional[float] = Field(default=None, examples=[0])
    weight_fn_test: Optional[float] = Field(default=None, examples=[-1])
    weight_tn_test: Optional[float] = Field(default=None, examples=[0.35])
    weight_tp_test: Optional[float] = Field(default=None, examples=[0])

class SimulationResponse(BaseModel):
    threshold_train: Optional[float] = Field(default=None,examples=[49])
    tp_train: Optional[float] = Field(default=None,examples=[644])
    fp_train: Optional[float] = Field(default=None,examples=[164])
    fn_train: Optional[float] = Field(default=None,examples=[16])
    tn_train: Optional[float] = Field(default=None,examples=[176])
    total_cost_train: Optional[float] = Field(default=None,examples=[1260])
    pnl_selected_train: Optional[float] = Field(default=None,examples=[1024000000])
    profit_loss_curve_train: dict[str, list] = Field(
        examples=[{
            "x_risk_threshold_train": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
            "y_pnl_train": [
                1025600000, 1024000000, 1020800000, 1017600000, 1014400000, 1011200000,
                1008000000, 1004800000, 1001600000, 998400000, 995200000, 992000000,
                988800000, 985600000, 982400000, 979200000, 976000000, 972800000,
                969600000, 966400000, 963200000, 960000000, 956800000, 953600000,
                950400000, 947200000, 944000000, 940800000, 937600000, 934400000,
                931200000, 928000000, 924800000, 921600000, 918400000, 915200000,
                912000000, 908800000, 905600000, 902400000, 899200000, 896000000,
                892800000, 889600000, 886400000, 883200000, 880000000, 876800000,
                873600000, 870400000, 867200000, 864000000, 860800000, 857600000,
                854400000, 851200000, 848000000, 844800000, 841600000, 838400000,
                835200000, 832000000, 828800000, 825600000, 822400000, 819200000,
                816000000, 812800000, 809600000, 806400000, 803200000, 800000000,
                796800000, 793600000, 790400000, 787200000, 784000000, 780800000,
                777600000, 774400000, 771200000, 768000000, 764800000, 761600000,
                758400000, 755200000, 752000000, 748800000, 745600000, 742400000,
                739200000, 736000000, 732800000, 729600000, 726400000, 723200000,
                720000000
            ]
        }]
    )
    threshold_test: Optional[float] = Field(default=None,examples=[49])
    tp_test: Optional[float] = Field(default=None,examples=[644])
    fp_test: Optional[float] = Field(default=None,examples=[164])
    fn_test: Optional[float] = Field(default=None,examples=[16])
    tn_test: Optional[float] = Field(default=None,examples=[176])
    total_cost_test: Optional[float] = Field(default=None,examples=[1260])
    pnl_selected_test: Optional[float] = Field(default=None,examples=[1024000000])
    profit_loss_curve_test: dict[str, list] = Field(
        examples=[{
            "x_risk_threshold_test": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
            "y_pnl_test": [
                1025600000, 1024000000, 1020800000, 1017600000, 1014400000, 1011200000,
                1008000000, 1004800000, 1001600000, 998400000, 995200000, 992000000,
                988800000, 985600000, 982400000, 979200000, 976000000, 972800000,
                969600000, 966400000, 963200000, 960000000, 956800000, 953600000,
                950400000, 947200000, 944000000, 940800000, 937600000, 934400000,
                931200000, 928000000, 924800000, 921600000, 918400000, 915200000,
                912000000, 908800000, 905600000, 902400000, 899200000, 896000000,
                892800000, 889600000, 886400000, 883200000, 880000000, 876800000,
                873600000, 870400000, 867200000, 864000000, 860800000, 857600000,
                854400000, 851200000, 848000000, 844800000, 841600000, 838400000,
                835200000, 832000000, 828800000, 825600000, 822400000, 819200000,
                816000000, 812800000, 809600000, 806400000, 803200000, 800000000,
                796800000, 793600000, 790400000, 787200000, 784000000, 780800000,
                777600000, 774400000, 771200000, 768000000, 764800000, 761600000,
                758400000, 755200000, 752000000, 748800000, 745600000, 742400000,
                739200000, 736000000, 732800000, 729600000, 726400000, 723200000,
                720000000
            ]
        }]
    )


class FeatureRecommendationRequest(BaseModel):
    table_name: str = Field(default=DUMMY_DATASET_TABLE_NAME, examples=[DUMMY_DATASET_TABLE_NAME])
    start_date: date = Field(default="1900-01-01", examples=["1900-01-01"])
    end_date: date = Field(default="2100-01-01", examples=["2100-01-01"])
    selected_risk_definition: str = Field(examples=['Simple'])
    risk_definition: Union[dict, str] = Field(examples=["dpd_m1"])
    min_iv: float = Field(default=0.02, examples=[0.02])
    max_iv: float = Field(default=0.5, examples=[0.5])
    min_qs: float = Field(default=0.01, examples=[0.001])

    @field_validator("risk_definition")
    def ensure_dict_or_str(cls, v):
        if isinstance(v, dict):
            return v  # accept dict
        return str(v)  # If the value is not dict, cast to string

class FeatureRecommendationResponse(BaseModel):
    dataset_table_name: str = Field(examples=[DUMMY_DATASET_TABLE_NAME])
    dataset_start_date: date = Field(examples=["2000-01-01"])
    dataset_end_date: date = Field(examples=["2050-12-31"])
    features: list[str] = Field(examples=[['age', 'province', 'income_class', 'income_type', 'education', 'marital_status']])

class FeatureModelResponse(BaseModel):
    schema_features: dict = Field(
        examples=[
            {
                "age": "int",
                "occupation": "object",
            }
        ]
    ),
    unique_values_features: dict[str, Union[dict[str, int], dict[str, float], list[str]]] = Field(
        examples=[
            {
                "age": {
                    "min": 23,
                    "max": 55,
                },
                "marital_status": [
                    "MARRIED", "SINGLE", "WIDOW", "DIVORCED"
                ]
            }
        ]
    )
