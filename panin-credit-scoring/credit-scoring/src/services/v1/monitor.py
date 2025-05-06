import datetime
import logging
import mlflow
import numpy as np
import pandas as pd
import uuid
from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound
from sqlalchemy import create_engine
from fastapi import HTTPException, status

import src.crud.v1.score as crud_score
import src.services.v1.model as model_svc
from src.schemas.v1.predict import MonitorHistoryBase

from src.config.constant import MODEL_STATUS_DEPLOYED, TYPE_CATEGORICAL
from src.config.constant import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.config.constant import SQLALCHEMY_DATABASE_URI

logger = logging.getLogger(__name__)

def get_all_monitoring_history(db: Session, model_name: str):
    # Update drift metric on demand
    try:
        # always update current month metric
        _ = update_latest_drift_metric(db, model_name)
    except Exception:
        logger.exception('update latest monitoring metric failed')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail='internal server error')
    
    result = crud_score.get_monitoring_history(db, model_name)
    return result 

def get_latest_monitoring_history(db: Session, model_name: str, n_months: int):
    # Update drift metric on demand
    try:
        # always update current month metric
        _ = update_latest_drift_metric(db, model_name)
    except Exception:
        logger.exception('update latest monitoring metric failed')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail='internal server error')
   
    result = crud_score.get_latest_monitoring_history(db, model_name, n_months)
    return result 

def update_latest_drift_metric(db: Session, model_name: str):
    try:
        logger.info(f'get model [{model_name}] detail from db')
        model_detail = model_svc.get_model_by_name(db, model_name)
    except NoResultFound:
        logger.exception(f'model [{model_name}] does not exist')
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'model [{model_name}] does not exist')
    
    if model_detail.status != MODEL_STATUS_DEPLOYED:
        msg = f'model [{model_name}] has not been deployed'
        logger.warning(msg)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=msg)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    artifact_uri = f'runs:/{model_detail.run_id}'
    score_bin_metadata = mlflow.artifacts.load_dict(artifact_uri + "/score_bin_metadata.json")
    input_bin_metadata = mlflow.artifacts.load_dict(artifact_uri + "/input_bin_metadata.json")

    # TODO: use sqlalchemy instead of pandas, move to crud layer
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    data = pd.read_sql_table('scoring_result', engine, 
                             columns=['total_score', 'input_detail', 'scoring_model_name', 'predicted_at'])

    # step 1: get last monitoring date
    # step 2: loop until current month
    # step 3: update the monitoring metrics of the month
    last_monitored_at = crud_score.get_latest_monitornig_date(db, model_name)[0]
    this_month_monitored_at = datetime.datetime.now().replace(day=25, hour=0, minute=0, second=0, microsecond=0)

    logger.info(f'Last monitored at: {last_monitored_at}')
    logger.info(f'This month monitored at: {this_month_monitored_at}')

    monitor_dt_queue = []

    if last_monitored_at:
        delta_monitored_at = this_month_monitored_at - last_monitored_at
        logger.info(f'Monitoring time delta: {delta_monitored_at}')

        if delta_monitored_at.days > 32:
            monitor_period = last_monitored_at

            while monitor_period <= this_month_monitored_at:
                monitor_period += datetime.timedelta(days=30)
                monitor_period = monitor_period.replace(day=25, hour=0, minute=0, second=0, microsecond=0)
                
                logger.info(f'Adding monitor period {monitor_period} to queue...')
                monitor_dt_queue.append(monitor_period)
    else:
        monitor_dt_queue.append(this_month_monitored_at) 

    for monitor_dt in monitor_dt_queue:
        logger.info(f'Calculating monitoring metrics for period: [{monitor_dt}]')
        try:
            this_month_dt = monitor_dt.replace(day=1, hour=0, minute=0, second=1)
            next_month_dt = this_month_dt + datetime.timedelta(days=32)
            next_month_dt = next_month_dt.replace(day=1, hour=0, minute=0, second=1)
            logger.info(f'Dataset start at: {this_month_dt}; end at: {next_month_dt}')

            monthly_data = data.loc[
                (data.scoring_model_name == model_name) & 
                (data.predicted_at >= this_month_dt) & 
                (data.predicted_at < next_month_dt)
            ].copy()

            if len(monthly_data) <= 0:
                continue

            logger.info('# Original input data sample #')
            logger.info(monthly_data.head().to_string())
        
            X_input = pd.DataFrame.from_records(monthly_data.input_detail.values)

            latest_scores = monthly_data['total_score'].values

            logger.info('# Input data sample #')
            logger.info(X_input.head().to_string())
      
            score_bin_edges = score_bin_metadata['score_bin_edges']

            reference_points = np.array(list(map(int, score_bin_metadata['score_bin_count'])))
            monitored_points, _ = np.histogram(latest_scores, bins=score_bin_edges)

            # output drift calculation
            psi = get_psi(reference_points, monitored_points)
            logger.info(f"Model [{model_name}] - Population Stability Index (PSI): {psi}")
        
            # input drift calculation
            input_drift_detail = {}
            
            for column in input_bin_metadata:
                mon_column = X_input[column]
                ref_bin_edges = input_bin_metadata[column]['input_bin_edges']
                ref_point = np.array(list(map(int, input_bin_metadata[column]['input_bin_count'])))
                if input_bin_metadata[column]['type'] == TYPE_CATEGORICAL:
                    category_to_index = input_bin_metadata[column]['category_to_index']
                    # Map categories to numeric values for both datasets
                    mon_embedded = mon_column.map(category_to_index).fillna(-1).astype(int)
                    mon_point, _ = np.histogram(mon_embedded, bins=ref_bin_edges, density=True)

                else:
                    # For numerical data, calculate bin edges using only reference data
                    mon_point, _ = np.histogram(mon_column, bins=ref_bin_edges, density=True)
                    
                # Normalize histograms to probability distributions
                ref_prob = ref_point / ref_point.sum() if ref_point.sum() != 0 else np.zeros_like(ref_point)
                mon_prob = mon_point / mon_point.sum() if mon_point.sum() != 0 else np.zeros_like(mon_point)

                ref_prob = np.where(ref_prob == 0, 1e-6, ref_prob)
                mon_prob = np.where(mon_prob == 0, 1e-6, mon_prob)
                
                # Compute KL divergence
                kl = get_kl_divergence(ref_prob, mon_prob)
                significant_change = kl > 0.05 # More than threshold
                significant_change = bool(significant_change)

                type = input_bin_metadata[column]['type']

                # Store results in dictionary
                input_drift_detail[column] = {"kl": kl, "is_significant": significant_change, "type": type}

            kl_values = [result['kl'] for result in input_drift_detail.values()]
            kl_mean = round(np.mean(kl_values), 2)
            kl_median = round(np.median(kl_values), 2)

            logger.info(f"Model [{model_name}] - Mean Kullback-Leibler divergence (KL): {kl_mean}")
            logger.info(f"Model [{model_name}] - Median Kullback-Leibler divergence (KL): {kl_median}")
            logger.info(f"Model [{model_name}] - Detail Kullback-Leibler divergence (KL): {input_drift_detail}")

            monitor_id = str(uuid.uuid4())
            # lock monitor date to 25
            # monitor_datetime = datetime.datetime.now().replace(day=25, hour=0, minute=0, second=0, microsecond=0)

            logger.info("store monitoring history into db")
            monitor_history = MonitorHistoryBase(
                monitor_id=monitor_id,
                monitored_at=monitor_dt,
                scoring_model_name=model_name,
                psi = psi,
                kl_mean = kl_mean,
                kl_median = kl_median,
                input_drift_detail = input_drift_detail,
                num_account = len(latest_scores),
                min_score = round(min(latest_scores), 2),
                max_score = round(max(latest_scores), 2), 
                avg_score = round(np.mean(latest_scores), 2),
                median_score = round(np.median(latest_scores), 2),
                std_score = round(np.std(latest_scores), 2)
            )
            _ = crud_score.upsert_monitoring_history(db, monitor_history)
        except Exception:
            logger.exception('update monitoring metircs failed')
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='internal server error')

    return True

def get_psi(reference_points: np.array, monitored_points: np.array):
    reference_prop = reference_points / sum(reference_points)
    monitored_prop = monitored_points / sum(monitored_points)

    reference_prop = np.where(reference_prop == 0, 1e-6, reference_prop)
    monitored_prop = np.where(monitored_prop == 0, 1e-6, monitored_prop)

    psi = np.sum((monitored_prop - reference_prop) * np.log(monitored_prop / reference_prop))
    psi = round(psi, 2)
    logger.info(f"Population Stability Index (PSI): {psi}")

    return psi

def get_kl_divergence(p, q):
        '''
        Calculate KL divergence between two probability distributions without external libraries.
        Args:
        - p: Array-like, probability distribution for p.
        - q: Array-like, probability distribution for q.
        Returns:
        - KL divergence value.
        '''
        epsilon = 1e-10  # Small constant to avoid log(0) and division by zero
        
        # Replace zero values in p and q with epsilon (anti-zero probability)
        p = np.where(p == 0, epsilon, p)
        q = np.where(q == 0, epsilon, q)
        
        # Ensure p and q are still valid probability distributions (between epsilon and 1)
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)

        # Calculate KL divergence: Sum of p * log(p / q)
        return round(np.sum(p * np.log(p / q)), 2)