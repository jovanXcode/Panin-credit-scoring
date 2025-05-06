import mlflow 
import logging
from fastapi import HTTPException, status
from mlflow import MlflowClient
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound

import src.services.v1.model as model_svc

from src.crud.v1 import model as model_crud
from src.schemas.v1.deployment import DeploymentRequest, DeploymentResponse
from src.schemas.v1.deployment import UndeploymentResponse

from src.config.constant import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.config.constant import MODEL_STATUS_DEPLOYED, MODEL_STATUS_TRAINED
from src.config.constant import STATUS_SUCCESS

logger = logging.getLogger(__name__)

def deploy_model(db: Session, req: DeploymentRequest):
    is_updated = False

    # get model detail from db
    try:
        logger.info(f'get model {req.deployed_model_name} from db')
        model_detail = model_svc.get_model_by_name(db, req.deployed_model_name)
    except NoResultFound:
        err_msg = f'Model [{req.deployed_model_name}] does not exist'
        logger.info(err_msg)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=err_msg)

    # validate if there is no model is being deployed in production
    # only 1 model can be deployed at a time
    models = model_crud.get_models_by_status(db, MODEL_STATUS_DEPLOYED)
    if len(models) > 0:
        logger.info(f'number of deployed models: {len(models)}')
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f'only 1 model can be deployed at a time, please undeploy model [{models[0].name}] first')

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        model_uri = f"runs:/{model_detail.run_id}/model"
        logger.info(f'model_uri: {model_uri}')

        # register model on mlflow
        mlflow.register_model(model_uri, model_detail.name)

        # update model status to DEPLOYED in db
        is_updated = model_crud.update_model_status(db, model_name=req.deployed_model_name, 
                                                    status=MODEL_STATUS_DEPLOYED)
    
        # TODO: FIX REGISTERED MODEL ALIAS
        # create "champion" alias for version 1 of model "example-model"
        client = MlflowClient()

        client.set_registered_model_alias(model_detail.name, alias='deployed', version="1")
        client.set_registered_model_tag(model_detail.name, "task", "scoring")
        logger.info('set registered model alias to deployed success!')
    except Exception as e:
        logger.exception('deploy model error')

        # revert model status
        if is_updated:
            _ = model_crud.update_model_status(db, model_name=req.deployed_model_name, 
                                               status=model_detail.status)
            
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail='deploy model error')

    response = DeploymentResponse(
        deployed_model_name = model_detail.name,
        deployed_model_run_id = model_detail.run_id,
        deploy_status = STATUS_SUCCESS
    )
    return response

def undeploy_model(db: Session, req: DeploymentRequest):
    is_updated = False

    # get model detail from db
    try:
        logger.info(f'get model {req.deployed_model_name} from db')
        model_detail = model_svc.get_model_by_name(db, req.deployed_model_name)
    except NoResultFound:
        err_msg = f'Model [{req.deployed_model_name}] does not exist'
        logger.info(err_msg)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err_msg)
    
    if model_detail.status != MODEL_STATUS_DEPLOYED:
        err_msg = f"The Model Hasn't Been Deployed. Current Status is [{model_detail.status}]"
        logger.info(err_msg)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=err_msg)
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        model_uri = f"runs:/{model_detail.run_id}/model"
        logger.info(f'model_uri: {model_uri}')

        # delete registered model from mlflow
        client = MlflowClient()
        client.delete_registered_model(name=req.deployed_model_name)

        # update model status on db
        is_updated = model_crud.update_model_status(db, req.deployed_model_name, MODEL_STATUS_TRAINED)
    except Exception as e:
        logger.exception('undeploy model error')

        # revert model status
        if is_updated:
            model_crud.update_model_status(db, model_name=req.deployed_model_name, 
                                            status=model_detail.status)
            
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail='undeploy model error')
    
    response = UndeploymentResponse(
        undeployed_model_name = model_detail.name,
        undeployed_model_run_id = model_detail.run_id,
        undeploy_status = STATUS_SUCCESS
    )
    return response