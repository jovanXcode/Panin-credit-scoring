import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound

import src.services.v1.train as train_svc
from src.schemas.v1.train import FeatureModelResponse
from src.utils.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/v1/features',
    tags=['features'],
    responses={
        status.HTTP_404_NOT_FOUND: {"message": "Not found"}, 
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"message": "Unprocessable entity"}, 
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"message": "Server error"},
    },
)

@router.get("/{model_name}", response_model=FeatureModelResponse,
            summary='get features by model name')
async def get_feature_metadata(model_name: str, db: Session = Depends(get_db)):
    try: 
        result = train_svc.get_features_by_model_name(model_name, db)
    except NoResultFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model [{model_name}] not found")

    return result