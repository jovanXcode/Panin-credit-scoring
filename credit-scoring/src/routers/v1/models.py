import logging
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound

from src.config.constant import MODEL_STATUS_TRAINING
from src.schemas.v1.deployment import DeploymentRequest
from src.schemas.v1.deployment import DeploymentResponse, UndeploymentResponse
from src.schemas.v1.model import ModelResponse, AlgorithmListResponse, ModelListResponse
from src.schemas.v1.model import DeleteModelRequest, DeleteModelResponse
from src.schemas.v1.train import  FeatureRecommendationRequest, TrainRequest, FeatureRecommendationResponse
from src.schemas.v1.train import UpdateFailedTrainRequest, SimulationRequest, SimulationResponse
from src.schemas.v1.predict import PredictRequest, PredictResponse
from src.schemas.v1.predict import PredictHistoryListResponse
from src.schemas.v1.predict import MonitorHistoryListBase, AggregatedMonitorHistoryBase
import src.crud.v1.model as crud
import src.services.v1.dataset as dataset_svc
import src.services.v1.train as train_svc
import src.services.v1.model as model_svc
import src.services.v1.monitor as monitor_svc
from src.services.v1.deployment import deploy_model, undeploy_model
from src.services.v1.train import train_model, train_model_async
from src.services.v1.predict import predict_risk_score
from src.services.v1.predict import get_scoring_result, get_scoring_history 
from src.utils.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/v1/models',
    tags=['models'],
    responses={
        status.HTTP_404_NOT_FOUND: {"message": "Not found"}, 
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"message": "Server error"}
    },
)

@router.get("/algorithms", response_model=AlgorithmListResponse, 
             summary='get algorithm choices')
async def get_algorithms(db: Session = Depends(get_db)):
    algorithms =  model_svc.get_algorithms(db)
    return {
        'algorithms': algorithms
    }

@router.post("/deploy", response_model=DeploymentResponse,
             summary='deploy a trained model, only one model can be deployed in production')
async def deploy(req: DeploymentRequest, db: Session = Depends(get_db)):
    return deploy_model(db, req)

@router.post("/undeploy", response_model=UndeploymentResponse,
             summary='undeployed a deployed model')
async def undeploy(req: DeploymentRequest, db: Session = Depends(get_db)):
    return undeploy_model(db, req)

@router.get("/", response_model=ModelListResponse,
            summary='get all models')
async def get_all_models(db: Session = Depends(get_db)):
    models = model_svc.get_models(db) 
    return {
        'models': models
    }

@router.delete("/", response_model=DeleteModelResponse,
               summary='delete a model')
async def delete_model(req: DeleteModelRequest, db: Session = Depends(get_db)):
    is_deleted = model_svc.delete_model(db, req.scoring_model_name)
    return DeleteModelResponse(is_deleted = is_deleted)

@router.get("/{model_name}", response_model=ModelResponse,
            summary='get model detail by model name')
async def get_model_detail(model_name: str, db: Session = Depends(get_db)):
    try: 
        result =  model_svc.get_model_by_name(db, model_name)
    except NoResultFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model [{model_name}] not found")

    return result

# automate feature selection
@router.post("/train/feature/recommendation", status_code=status.HTTP_200_OK, response_model=FeatureRecommendationResponse,
             summary='Automate feature selection for model training')
async def feature_recommendation(req: FeatureRecommendationRequest, db: Session = Depends(get_db)):
    """
    Endpoint that allows users to request automated feature selection based on min_iv, max_iv, and min_qs query parameters.
    """
    dataset_ = dataset_svc.get_dataset_by_name(db, req.table_name)
    if not dataset_:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"Dataset [{req.table_name}] does not exist, please create the dataset first")

    # Validate min_iv and max_iv
    if req.min_iv >= req.max_iv:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="min_iv should be less than max_iv")

    result = train_svc.get_feature_recommendation(db, req=req)
    
    return result

@router.post("/train", status_code=status.HTTP_201_CREATED, response_model=ModelResponse,
             summary='create and train a new model')
def train(req: TrainRequest, db: Session = Depends(get_db)):
    try:
        dataset_ = dataset_svc.get_dataset_by_name(db, req.table_name)
        if not dataset_:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                                detail=f"Dataset [{req.table_name}] does not exist, please create the dataset first")
            
        model_ = model_svc.get_model_by_name(db, req.scoring_model_name)
        if model_.status != MODEL_STATUS_TRAINING:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                                detail=f"Model [{req.scoring_model_name}] already created, please use another unique model_name")
    except NoResultFound:
        pass

    new_model = train_model(db, req)

    return new_model

@router.post("/train_async", status_code=status.HTTP_201_CREATED,
             summary='create and train a new model, return task_id to track the training progress')
async def train_async(req: TrainRequest, request: Request, db: Session = Depends(get_db)):
    try:
        dataset_ = dataset_svc.get_dataset_by_name(db, req.table_name)
        if not dataset_:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                                detail=f"Dataset [{req.table_name}] does not exist, please create the dataset first")
        
        model_ = model_svc.get_model_by_name(db, req.scoring_model_name)
        if model_:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                                detail=f"Model [{req.scoring_model_name}] already created, please use another unique model_name")
    except NoResultFound:
        pass
    
    auth_token = request.headers.get("Authorization")
    auth_token = auth_token.replace("Bearer ", "")
    
    result = train_model_async(db, req, auth_token)
    return result

@router.post("/train/failed/record/{scoring_model_name}", status_code=status.HTTP_200_OK,
             summary='Record failed train reason')
def record_failed_train(scoring_model_name: str, db: Session = Depends(get_db)):
    result = crud.get_train_status_by_model_name(db, scoring_model_name)
    return result

@router.get("/train/status/{task_id}",
            summary='get train status and result by task id')
async def get_train_async_status(task_id: int, db: Session = Depends(get_db)):
    try:
        task_result = crud.get_train_status_by_task_id(db, task_id)
        if not task_result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                                detail=f"Task result [{task_id}] does not exist, please check the task_id spell.")
    except NoResultFound:
        pass
    return task_result

@router.post("/simulation/{scoring_model_name}", status_code=status.HTTP_200_OK, response_model=SimulationResponse,
             summary='Get scoring model performance simulation given the threshold')
def model_simulation(req: SimulationRequest, db: Session = Depends(get_db)):
    result = train_svc.get_model_simulation(db, req)
    return result

@router.post("/threshold/{scoring_model_name}", status_code=status.HTTP_200_OK,
             summary='Set threshold of the selected model')
def threshold_model(scoring_model_name: str, threshold : float,  db: Session = Depends(get_db)):
    result = crud.update_threshold_by_model(db, scoring_model_name, threshold)
    return {'is_success': True}

# @router.get("/train/status", response_model=TrainStatusListResponse,
#             summary='get all train status and result')
# async def get_train_async_status(db: Session = Depends(get_db)):
#     result = get_all_train_status(db)
#     return {'result': result}

@router.post("/predict", response_model=PredictResponse,
             summary='trigger model scoring against the selected dataset')
async def predict(req: PredictRequest, db: Session = Depends(get_db)):
    try:
        _ = model_svc.get_model_by_name(db, req.scoring_model_name)
    except NoResultFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model [{req.scoring_model_name}] not found")

    result = predict_risk_score(db, req)
    return result

@router.get("/predict/history", response_model=PredictHistoryListResponse,
            summary='get list of predict history and detail')
async def get_predict_history(db: Session = Depends(get_db)):
    result = get_scoring_history(db)
    return {'result': result}

@router.get("/monitor/history/{model_name}", response_model=MonitorHistoryListBase,
            summary='get monitoring history by model name')
async def get_monitoring_history(model_name: str, db: Session = Depends(get_db)):
    result = monitor_svc.get_all_monitoring_history(db, model_name)
    return {'history': result}

@router.get("/monitor/history/summary/{model_name}/m/{n_months}", response_model=AggregatedMonitorHistoryBase,
            summary='get latest (aggregated) monitoring history')
async def get_latest_monitoring_history(model_name: str, n_months: int, db: Session = Depends(get_db)):
    result = monitor_svc.get_latest_monitoring_history(db, model_name, n_months)
    return {'history': result}

@router.get("/scoring/result", summary='')
def get_result(limit: int = 10):
    result = get_scoring_result(limit)
    return result

