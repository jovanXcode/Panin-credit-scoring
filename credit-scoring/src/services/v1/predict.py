import json
import datetime
import logging
import mlflow
import numpy as np
import pandas as pd
import uuid
import random
from sqlalchemy import dialects, String, Float, DateTime
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound

from fastapi import HTTPException, status

import src.services.v1.model as model_svc
import src.crud.v1.score as score_crud
from src.crud.v1.score import add_scoring_history, get_scoring_history

from src.schemas.v1.predict import PredictRequest, PredictResponse
from src.schemas.v1.predict import PredictHistoryBase, MonitorHistoryBase

from src.config.constant import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MIN_RISK_SCORE, MAX_RISK_SCORE
from src.config.constant import SQLALCHEMY_DATABASE_URI, MODEL_STATUS_DEPLOYED, DECISION_ACCEPT, DECISION_CHECK, DECISION_REJECT

logger = logging.getLogger(__name__)

def get_predict_history(db: Session):
    history = get_scoring_history(db)
    return history 

def get_scoring_result(db: Session, limit: int=10):
    engine = create_engine(SQLALCHEMY_DATABASE_URI)

    df = pd.read_sql_table('scoring_result', engine).head(limit)
    values = df.to_dict(orient='records') 
    keys = values[0].keys()

    logger.info(list(keys))
    return {
        'available_keys': list(keys),
        'results': values
    }

def filter_data_by_datetime(df_: pd.DataFrame, datatime_cols: list, start_date='1900-01-01', end_date='2100-01-01'):
    if len(datatime_cols) > 0:
        dt_ref_col = datatime_cols[0]
        if start_date and end_date:
            idx = (df_[dt_ref_col].dt.date >= start_date) & (df_[dt_ref_col].dt.date <= end_date)
        elif end_date:
            idx = df_[dt_ref_col].dt.date >= start_date
        elif end_date:
            idx = df_[dt_ref_col].dt.date <= end_date

        df_ = df_.loc[idx]

        data_len = len(df_)
        if data_len < 1:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=f'at least 1 data points to be scored, current: [{data_len}]')
        
        return df_
    
def preprocess_input(df_: pd.DataFrame, reference: pd.DataFrame):
    for col in df_.columns:
        if reference[col].dtype == 'object':  # Jika tipe data adalah object
            df_[col] = df_[col].astype(str).str.upper()  # Ubah menjadi string dan kapital
        else:
            df_[col] = df_[col].astype(reference[col].dtype)  # Samakan tipe data lainnya

    return df_

def predict_risk_score(db: Session, req: PredictRequest):
    try:
        logger.info(f'get model [{req.scoring_model_name}] detail from db')
        model_detail = model_svc.get_model_by_name(db, req.scoring_model_name)
    except NoResultFound:
        logger.exception('model does not exist')
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'model [{req.scoring_model_name}] does not exist')
    
    if model_detail.status != MODEL_STATUS_DEPLOYED:
        msg = f'model [{req.scoring_model_name}] has not been deployed and now allowed to make prediction'
        logger.warning(msg)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=msg)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    selected_features = model_detail.features

    # Validate threshold
    try:
        threshold = model_detail.detail["threshold"]
    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Please set the threshold before making prediction."
        )

    # Set ID
    predict_id = str(uuid.uuid4())
    predict_datetime = datetime.datetime.now()
    # For testing purposes
    # predict_datetime = datetime.datetime.now() + datetime.timedelta(random.choice([1, -1]) * random.choice([i for i in range(1, 360)]))

    # Load model info
    artifact_uri = f'runs:/{model_detail.run_id}'
    input_mapping = mlflow.artifacts.load_dict(artifact_uri + "/model/input_example.json")
    df_based = pd.DataFrame(input_mapping['data'], columns=input_mapping['columns'])

    X = pd.DataFrame(req.input)
    X = X[selected_features]

    X = preprocess_input(X, df_based)

    logger.info(f'dataframe schema for input model: {X.dtypes}')

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    model = mlflow.sklearn.load_model(artifact_uri + '/model')
    
    logger.info('model scoring on the retrieved data')

    scores = model.score(X)

    point_mapping = mlflow.artifacts.load_dict(artifact_uri + "/point_mapping.json")
    
    df_feature_score = model.binning_process.transform(X, metric='indices')
    logger.info("BEFORE BINNING TRANSFORMATION")
    logger.info(df_feature_score.head(3))

    logger.info(point_mapping.keys())
    logger.info(point_mapping[df_feature_score.columns[0]])

    for col in df_feature_score.columns:
        df_feature_score[col] = df_feature_score[col].astype(str).map(point_mapping[col])

    logger.info("AFTER BINNING TRANSFORMATION")
    logger.info(df_feature_score.head(3))

    logger.info("binning metadata")
    logger.info(model.binning_process)
    
    total_scores = df_feature_score.sum(axis=1).tolist()

    input_detail = X.copy()
    output_detail = pd.DataFrame({
        'total_score': np.round(scores,2)
    })

    # Add result to response
    total_scores = round(float(total_scores[0]),2)
        
    column_names = df_feature_score.columns.tolist()

    score_breakdown = {
        col: round(val, 2) if isinstance(val, float) and not pd.isna(val) else None
        for col, val in df_feature_score.iloc[0].items()
    }

    logger.info("Score breakdown after removing NaN values:")
    logger.info(score_breakdown)

    # Recommended limit calculation
    sc_model = model.estimator_
    X_binned = model.binning_process_.transform(X)
    # Non-default proba
    y_pred_proba = sc_model.predict_proba(X_binned)[:, 0]

    # 2 (40% from default probability) and 3 (60% from the risk score)
    adjusted_risk_score = (y_pred_proba * 2 + scores / 100 * 3) / (2 + 3)
    adjusted_risk_score = adjusted_risk_score.round(3)
    logger.info(f'non-default probability: {y_pred_proba}')
    logger.info(f'estimated risk score: {scores}')
    logger.info(f'adjusted risk score: {adjusted_risk_score}')
    
    recommend_limit = req.max_limit * adjusted_risk_score
     
    recommend_limit = max(recommend_limit, req.min_limit)

    # rounding to the closes 100.000 nominal
    # example: 1.561.123 -> 1.500.000
    # example: 10.211.394 -> 10.200.000
    recommend_limit = (recommend_limit // 1e5) * 1e5

    # LOGGING FOR DEBUGGING DECISION
    logger.info(f"Raw Score: {scores}, Threshold: {threshold}, MAX_RISK_SCORE: {MAX_RISK_SCORE}")

    if total_scores > threshold:
        if total_scores > int(((MAX_RISK_SCORE - threshold) / 2) + threshold):
            recommend_decision = DECISION_ACCEPT
        else:
            recommend_decision = DECISION_CHECK
    else:
        recommend_decision = DECISION_REJECT

    result_output = [
        {
            'score':total_scores,
            'default_probability': round(1 - y_pred_proba[0], 3),
            'detail':score_breakdown, 
            'recommended_limit' : round(float(recommend_limit), 2), 
            'recommended_decision' : recommend_decision
        }
    ]

    app_detail = pd.DataFrame([req.app_detail])

    # TODO: raise exception if n_row<=0
    n_rows = store_scoring_result(model_name=model_detail.name,
                          model_run_id=model_detail.run_id,
                          predict_id=predict_id,
                          app_id=req.app_id,
                          predicted_at=predict_datetime,
                          app_detail= app_detail,
                          input_detail=input_detail,
                          output_detail=output_detail,
                          score_breakdown=df_feature_score,
                          total_score=total_scores)

    min_score = scores.min()
    max_score = scores.max()
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    median_score = np.quantile(scores, 0.5)
    data_size = len(scores)

    predict_detail = {
        'min_score': round(min_score, 2),
        'max_score': round(max_score, 2),
        'avg_score': round(avg_score, 2),
        'std_score': round(std_score, 2),
        'median_score': round(median_score, 2),
        'data_size': data_size 
    }

    logger.info("store predict history to db")
    logger.info(f"store predict history input to db : {req.input}")
    logger.info(f"store predict history output to db : {result_output}")

    predict_history = PredictHistoryBase(
        predict_id=predict_id,
        app_id=req.app_id,
        scoring_model_name=req.scoring_model_name,
        app_detail = req.app_detail,
        output_table_name='scoring_result',
        input=req.input,
        result = result_output,
        detail=predict_detail,
        predicted_at=predict_datetime
    ) 
    _ = add_scoring_history(db, predict_history)

    response = PredictResponse(
        predict_id=predict_id,
        app_id=req.app_id,
        app_detail = req.app_detail,
        psi=-1,
        kl_mean = -1,
        kl_median = -1,
        data_size = data_size,
        result = result_output,
        result_status='success',
        predicted_at=predict_datetime,
        min_risk_score = MIN_RISK_SCORE,
        max_risk_score = MAX_RISK_SCORE,
        range=[0, 25, 50, 75, 100]
    )
    return response

def store_scoring_result(model_name: str, model_run_id: str, predict_id:str, app_id: str, app_detail:dict,
                 predicted_at: datetime.datetime, input_detail, 
                 output_detail, score_breakdown, total_score 
                 ):

    app_detail = json.loads(app_detail.to_json(orient='records'))
    input_detail = json.loads(input_detail.to_json(orient='records'))
    output_detail = json.loads(output_detail.to_json(orient='records'))
    score_breakdown = json.loads(score_breakdown.to_json(orient='records'))
    

    result = {
        'app_detail' : app_detail,
        'input_detail': input_detail,
        'output_detail': output_detail,
        'score_breakdown': score_breakdown,
        'total_score': total_score 
    }

    df_result = pd.DataFrame(result)
    df_result['scoring_model_name'] = model_name
    df_result['scoring_model_run_id'] = model_run_id
    df_result['predict_id'] = predict_id
    df_result['app_id'] = app_id
    df_result['predicted_at'] = predicted_at
    #df_result['created_at'] = pd.to_datetime(datetime.datetime.now()).strftime('%Y-%m-%d %H:%M:%S')

    logger.info(df_result.dtypes)

    result_schema = {
        "input_detail":dialects.postgresql.JSONB,
        "output_detail":dialects.postgresql.JSONB,
        "score_breakdown":dialects.postgresql.JSONB,
        "total_score": Float,
        "scoring_model_name": String,
        "app_detail":dialects.postgresql.JSONB,
        "scoring_model_run_id": String,
        "predict_id": String,
        "app_id": String,
        "predicted_at": DateTime
    }

    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    rows = df_result.to_sql('scoring_result', engine, index=False, 
                            if_exists='append', dtype=result_schema)
    
    return rows



# SIDE NOTES
# =============================
# EXAMPLE: using version number
# model_name = "sk-learn-random-forest-reg-model"
# model_version = 1
# model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# model.predict(data)

# EXAMPLE: using version alias
# model_name = "sk-learn-random-forest-reg-model"
# alias = "champion"
# champion_version = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
# champion_version.predict(data)