import gc
import mlflow
import numpy as np
import pandas as pd
import tempfile
import uuid
import logging
from pathlib import Path
from datetime import datetime
from fastapi import HTTPException, status
import psutil
import subprocess
import threading
import re
import time
import json
from mlflow.pyfunc import PythonModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound
from optbinning import BinningProcess
from optbinning import Scorecard

from src.config.constant import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, API_TRAIN_TIMEOUT, SCORECARD_ELASTICNETCV, SCORECARD_ELASTICNET
from src.config.constant import SQLALCHEMY_DATABASE_URI, MODEL_STATUS_DEPLOYED,MIN_RISK_SCORE, MAX_RISK_SCORE
from src.config.constant import MODEL_STATUS_TRAINED, MODEL_STATUS_TRAINING, API_TRAIN_ENDPOINT, SCORECARD_LR, SCORECARD_LRCV
from src.config.constant import STATUS_FAILED, STATUS_PENDING, TYPE_CATEGORICAL, TYPE_NUMERICAL, STATUS_PROCESSING, STATUS_SUCCESS

import src.crud.v1.model as crud
import src.crud.v1.dataset as dataset_crud

from src.schemas.v1.train import TrainRequest, UpdateFailedTrainRequest, TrainStatusBase, SimulationRequest, SimulationResponse
from src.schemas.v1.train import FeatureRecommendationRequest, FeatureRecommendationResponse, FeatureModelResponse
from src.schemas.v1.model import ModelBase, ModelResponse

import src.services.v1.dataset as dataset_svc  
import src.services.v1.model as model_svc

logger = logging.getLogger(__name__)

def filter_data_by_datetime(df_: pd.DataFrame, datatime_cols: list, start_date='1900-01-01', end_date='2100-01-01'):
    if len(datatime_cols) > 0:
        dt_ref_col = datatime_cols[0]
        if start_date and end_date:
            idx = (df_[dt_ref_col].dt.date >= start_date) & (df_[dt_ref_col].dt.date <= end_date)
        elif start_date:
            idx = df_[dt_ref_col].dt.date >= start_date
        elif end_date:
            idx = df_[dt_ref_col].dt.date <= end_date

        df_ = df_.loc[idx]

        data_len = len(df_)
        if data_len < 100:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=f'at least 100 data points required for model training, given: [{data_len}]')
        
        return df_
    
def calculate_metrics(y_true, y_pred, y_pred_proba, scores):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * roc_auc - 1
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    for_metric = FN / (FN + TN) if (FN + TN) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    fdr = FP / (FP + TP) if (FP + TP) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0
    fnr = FN / (TP + FN) if (TP + FN) > 0 else 0
    lr_plus = recall / (tnr + 1e-6)
    lr_minus = fnr / (tnr + 1e-6)

    return {
        'min_score': round(scores.min(), 2),
        'max_score': round(scores.max(), 2),
        'avg_score': round(np.mean(scores), 2),
        'std_score': round(np.std(scores), 2),
        'median_score': round(np.quantile(scores, 0.5), 2),
        'data_size': int(len(y_true)),
        'tp': int(TP),
        'fp': int(FP),
        'fn': int(FN),
        'tn': int(TN),
        'accuracy': round(accuracy, 3),
        'f1_score': round(f1, 3),
        'roc_auc': round(roc_auc, 3),
        'gini_coefficient': round(gini, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'false_omission_rate': round(for_metric, 3),
        'negative_predictive_value': round(npv, 3),
        'false_discovery_rate': round(fdr, 3),
        'false_positive_rate': round(fpr, 3),
        'true_negative_rate': round(tnr, 3),
        'false_negative_rate': round(fnr, 3),
        'positive_likelihood_ratio': round(lr_plus, 3),
        'negative_likelihood_ratio': round(lr_minus, 3)
    }

def parse_feature_and_target(db, feature_cols: list[str], table_name: str):
    dataset_detail = dataset_crud.get_dataset_by_name(db, table_name)

    target_columns = ['dpd_m1'] 
    identity_columns = ['customer_id', 'loan_id']
    datetime_columns = ['datetime']

    if dataset_detail.target_column_names not in ['', None]:
        target_columns = dataset_detail.target_column_names

    logger.info(f'### TRAIN DATASET TARGET COLUMN: {target_columns}')

    if dataset_detail.identity_column_names not in [[], None]:
        identity_columns = dataset_detail.identity_column_names

    logger.info(f'### TRAIN DATASET IDENTITY COLUMNS: {identity_columns}')

    if dataset_detail.datetime_column_names not in [[], None]:
        datetime_columns = dataset_detail.datetime_column_names

    logger.info(f'### TRAIN DATASET DATETIME COLUMNS: {datetime_columns}')

    if feature_cols == []:
        table_info = dataset_svc.check_table(db, table_name)
        feature_cols = table_info['table_schema']
        feature_cols = list(feature_cols.keys())
        
    for col in identity_columns:
        if col in feature_cols:
            feature_cols.remove(col)

    for col in target_columns:
        if col in feature_cols:
            feature_cols.remove(col)

    for col in datetime_columns:
        if col in feature_cols:
            feature_cols.remove(col)

    logger.info(f'### TRAIN DATASET FEATURE COLUMNS: {feature_cols}')

    return feature_cols, target_columns, identity_columns, datetime_columns

def parse_target_definition(risk_definition: dict, df):
    """
    Recursively parse the risk_definition JSON into a condition for target).
    """
    # Initialize an empty series to hold the overall filter
    final_filter = None

    # Process each logical group ('and'/'or') in risk_definitions
    for operator, conditions in risk_definition.items():
        condition_group = None  # Initialize the condition group

        for col_name, rule in conditions.items():
            if col_name not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Column '{col_name}' not found in dataset."
                )
        
            # Generate the condition based on operator
            try:
                if rule['condition'] == "==":
                    condition = (df[col_name] == rule['value'])
                elif rule['condition'] == "!=":
                    condition = (df[col_name] != rule['value'])
                elif rule['condition'] == ">":
                    condition = (df[col_name] > rule['value'])
                elif rule['condition'] == "<":
                    condition = (df[col_name] < rule['value'])
                elif rule['condition'] == ">=":
                    condition = (df[col_name] >= rule['value'])
                elif rule['condition'] == "<=":
                    condition = (df[col_name] <= rule['value'])
                else:
                    raise ValueError(f"Unsupported condition: {rule['condition']}")
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error parsing condition for column '{col_name}': {str(e)}"
                )
            
            # Combine conditions within the same logical group
            if condition_group is None:
                condition_group = condition
            else:
                if operator == "and":
                    condition_group &= condition  # Logical AND for 'and'
                elif operator == "or":
                    condition_group |= condition  # Logical OR for 'or'

        # Combine groups together
        if final_filter is None:
            final_filter = condition_group
        else:
            final_filter |= condition_group  # Combine groups using OR as default

        # Apply the filter and add a new column 'filtered'
        y = final_filter.astype(int)

    return y

def parse_risk_definition(risk_definition: dict):
    """
    Recursively parse the risk_definition JSON into a condition for detail).
    """
    # Initialize a list to hold condition strings
    condition_strings = []

    # Iterate through the risk definition dictionary
    for operator, conditions in risk_definition.items():
        # Gather conditions for the current operator
        condition_parts = []
        last_operator = None 
        
        for real_condition, value in conditions.items():
            last_operator = operator  
            # Create string representation for the condition
            condition_str = f"{real_condition} {value['condition']} {repr(value['value'])}"
            condition_parts.append(f"({condition_str})")
        
        # Join conditions with the operator
        joined_conditions = f" {operator} ".join(condition_parts)
        condition_strings.append(joined_conditions)

    # Combine all condition strings into a final expression
    if condition_strings:
            combined_conditions = f" {last_operator} ".join(condition_strings)
    return combined_conditions

def train_model(db: Session, req: TrainRequest):
    train_start_at = time.perf_counter()
    model_id = str(uuid.uuid4())
    created_at = datetime.now()
    if isinstance(req.algorithm_id, uuid.UUID):
        algorithm_id = req.algorithm_id  
    else:
        algorithm_id = uuid.UUID(req.algorithm_id)

    feature_columns, target_columns, _, datetime_columns = parse_feature_and_target(db, req.features, req.table_name)
    cs_s = 10  # [1.00000000e-04 7.74263683e-04 5.99484250e-03 4.64158883e-02, 3.59381366e-01 2.78255940e+00 2.15443469e+01 1.66810054e+02, 1.29154967e+03 1.00000000e+04]
    l1_ratios_s= np.linspace(0.1, 1.0, 10)
    scoring = 'f1'

    logger.info(f'read dataset from table: {req.table_name}')
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    df = pd.read_sql_table(req.table_name, engine)

    if any(col in df.columns for col in datetime_columns):
        df = filter_data_by_datetime(df, datetime_columns, req.start_date, req.end_date)

    df = df.dropna(axis=1, how="all")

    # Sort dataframe by datetime to ensure temporal order
    date_col = datetime_columns[0]
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    X = df.copy()[feature_columns].reset_index(drop=True)
    
    if isinstance(req.risk_definition, dict):
        y = parse_target_definition(req.risk_definition, df)
        condition_string = parse_risk_definition(req.risk_definition)
    else:
        if req.risk_definition not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Risk definition column '{req.risk_definition}' not found in dataset. "
            )
        
        y = df[req.risk_definition].copy()
        y = np.where(y > 1, 1, 0)
        condition_string = req.risk_definition

    req.features = X.columns.tolist()

    # Updated test split logic
    if req.test_ratio > 0:
        logger.info(f"Using test_ratio={req.test_ratio} (prioritized)")
        split_idx = int(len(df) * (1 - req.test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Get and format date ranges
        train_dates = df.iloc[:split_idx][datetime_columns[0]]
        test_dates = df.iloc[split_idx:][datetime_columns[0]]

    elif req.test_n_month > 0:
        logger.info(f"test_ratio not valid (test_ratio={req.test_ratio}). Using test_n_month={req.test_n_month}")
        threshold = df[datetime_columns[0]].max() - pd.DateOffset(months=req.test_n_month - 1)
        threshold = threshold.replace(day=1)
        test_mask = df[datetime_columns[0]] >= threshold
        X_train, X_test = X[~test_mask], X[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]

        # Get and format date ranges
        train_dates = df[~test_mask][datetime_columns[0]]
        test_dates = df[test_mask][datetime_columns[0]]
        
    else:
        raise HTTPException(status_code=422, detail="Either a positive test_ratio or a positive test_n_month must be provided.")

    # Extract date ranges after the if-else clause 
    train_start_date = train_dates.min()
    train_end_date = train_dates.max()
    test_start_date = test_dates.min()
    test_end_date = test_dates.max()

    formatted_train_start = train_start_date.isoformat()
    formatted_train_end = train_end_date.isoformat()
    formatted_test_start = test_start_date.isoformat()
    formatted_test_end = test_end_date.isoformat()

    # Log date ranges
    logger.info(f"Training data covers period: {formatted_train_start} to {formatted_train_end}")
    logger.info(f"Testing data covers period: {formatted_test_start} to {formatted_test_end}")

    # Ensure minimum training size
    if len(X_train) < 100:
        raise HTTPException(status_code=422, detail=f"Insufficient training data: {len(X_train)} rows")

    # # Simplified test split logic
    # if req.test_ratio > 0:
    #     logger.info(f"Using test_ratio={req.test_ratio} (prioritized)")
    #     split_idx = int(len(df) * (1 - req.test_ratio))
    #     train_mask = df.index < split_idx
    # else:
    #     if req.test_n_month <= 0:
    #         raise HTTPException(status_code=422, detail="Either a positive test_ratio or a positive test_n_month must be provided.")
    #     logger.info(f"test_ratio not valid (test_ratio={req.test_ratio}). Using test_n_month={req.test_n_month}")
    #     threshold = df[datetime_columns[0]].max() - pd.DateOffset(months=req.test_n_month - 1)
    #     threshold = threshold.replace(day=1)
    #     train_mask = df[datetime_columns[0]] < threshold

    # X_train, X_test = X[train_mask], X[~train_mask]
    # y_train, y_test = y[train_mask], y[~train_mask]

    # train_dates = df[train_mask][datetime_columns[0]]
    # test_dates = df[~train_mask][datetime_columns[0]]
    # train_start_date = train_dates.min()
    # train_end_date = train_dates.max()
    # test_start_date = test_dates.min()
    # test_end_date = test_dates.max()

    # formatted_train_start = train_start_date.isoformat()
    # formatted_train_end = train_end_date.isoformat()
    # formatted_test_start = test_start_date.isoformat()
    # formatted_test_end = test_end_date.isoformat()

    # logger.info(f"Training data covers period: {formatted_train_start} to {formatted_train_end}")
    # logger.info(f"Testing data covers period: {formatted_test_start} to {formatted_test_end}")

    del df
    gc.collect()

    num_cols = X_train.select_dtypes(exclude='object').columns.tolist()
    cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()
    all_cols = X_train.columns.tolist()

    logger.info('start binning process...')
    binning_process = BinningProcess(variable_names=all_cols,
                    categorical_variables=cat_cols)
    binning_process.fit(X_train, y_train)

    X_train_binned = binning_process.transform(X_train)
    X_test_binned = binning_process.transform(X_test)

    df_binning_summary = binning_process.summary()
    df_binning_summary['feature_importance'] = df_binning_summary['iv'] / df_binning_summary['iv'].sum()
    logger.info('binning process done!')

    if isinstance(req.algorithm_id, uuid.UUID):
        algorithm_id = req.algorithm_id  
    else:
        algorithm_id = uuid.UUID(req.algorithm_id)

    try:
        algorithm_obj = crud.get_algorithm_by_id(db, algorithm_id)
        algorithm_name = algorithm_obj.name
    except Exception as e:
        logger.exception(f"Invalid algorithm_id provided: {algorithm_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Algorithm ID [{algorithm_id}] not found"
        )

    if algorithm_name == SCORECARD_LR:
        model = LogisticRegression(fit_intercept=False)
    elif algorithm_name == SCORECARD_LRCV:
        model = LogisticRegressionCV(cv=5, scoring=scoring, l1_ratios=l1_ratios_s, Cs = cs_s) 
    elif algorithm_name == SCORECARD_ELASTICNETCV:
        model = LogisticRegressionCV(penalty='elasticnet', solver='saga', l1_ratios=l1_ratios_s, cv=5, scoring=scoring, Cs = cs_s)
    elif algorithm_name == SCORECARD_ELASTICNET:
        model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, fit_intercept=False)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    lr_params = model.get_params()

    logger.info('build scorecard model...')
    scorecard = Scorecard(binning_process=binning_process,
                            estimator=model, scaling_method="min_max",
                            scaling_method_params={"min": MIN_RISK_SCORE, "max": MAX_RISK_SCORE})
    scorecard.fit(X_train, y_train)
    logger.info(scorecard.information(print_level=2))

    scorecard_table = scorecard.table(style="detailed")
    logger.info(f'\n{scorecard_table.to_string(index=False)}')
    logger.info('build scorecard model success!')

    sc_cols =  ['Variable', 'Bin id', 'Points']
    point_mapping = dict()
    scorecard_table = scorecard.table(style="detailed")
    for var in scorecard_table.Variable.unique():
        var_binning = scorecard_table.loc[scorecard_table.Variable==var, sc_cols[1:]]
        point_mapping[var] =  var_binning.set_index(sc_cols[1]).to_dict()['Points']
 
    # Feature weight importance
    df_sc_table = scorecard_table[['Variable', 'Points']].copy()
    df_sc_table['Points'] = df_sc_table['Points'].abs()

    # Extract feature points and variables
    feat_points = df_sc_table.groupby('Variable')['Points'].max()
    variables = feat_points.index

    # Compute contributions
    total_points = feat_points.sum()
    contributions = feat_points / total_points

    # Format the output
    feat_importance = {
        'Points': dict(zip(variables, np.round(feat_points.values, 2))),
        'Contribution': dict(zip(variables, np.round(contributions.values, 2))),
    }

    # Train predictions and scores
    score_train = scorecard.score(X_train)
    y_pred_train = scorecard.estimator_.predict(X_train_binned)
    y_pred_proba_train = scorecard.estimator_.predict_proba(X_train_binned)[:, 1]

    # Test predictions and scores
    score_test = scorecard.score(X_test)
    y_pred_test = scorecard.estimator_.predict(X_test_binned)
    y_pred_proba_test = scorecard.estimator_.predict_proba(X_test_binned)[:, 1]

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_pred_train, y_pred_proba_train, score_train)
    test_metrics = calculate_metrics(y_test, y_pred_test, y_pred_proba_test, score_test)

    _, score_bin_edges = np.histogram(score_test, bins="doane")
    score_bin_count, _ = np.histogram(score_test, bins=score_bin_edges)
    score_bin_metadata = {
        'score_bin_edges': list(score_bin_edges),
        'score_bin_count': list(score_bin_count)
    }
    # -------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------

    # Input drift reference optimization
    # Sort columns alphabetically to ensure consistent order
    sorted_columns = X.columns.sort_values()

    # Dictionary to store the histogram and bin edges for each column
    input_bin_metadata = {}

    # Iterate through columns and process
    for column in sorted_columns:
        ref_column = X[column]
        dtype_ = ref_column.dtype.name

        if dtype_ in ['object', 'category']:  # Categorical data
            unique_values = ref_column.dropna().unique()
            category_to_index = {category: idx for idx, category in enumerate(unique_values)}

            # Map categories to numeric values
            ref_embedded = ref_column.map(category_to_index).fillna(-1).astype(int)

            # Calculate histogram for categorical data
            input_bin_count, input_bin_edges = np.histogram(ref_embedded, bins="doane", density=True)

            # Store metadata for categorical columns
            input_bin_metadata[column] = {
                "input_bin_edges": input_bin_edges.tolist(),
                "input_bin_count": input_bin_count.tolist(),
                "type": TYPE_CATEGORICAL,
                "category_to_index": category_to_index
            }

        else:  # Numerical data
            # Calculate histogram for numerical data
            input_bin_count, input_bin_edges = np.histogram(ref_column.dropna(), bins="doane", density=True)

            # Store metadata for numerical columns
            input_bin_metadata[column] = {
                "input_bin_edges": input_bin_edges.tolist(),
                "input_bin_count": input_bin_count.tolist(),
                "type": TYPE_NUMERICAL
            }

    # qcut_labels = [f'qbin{qi}' for qi in list(range(1, 11, 1))]
    # qcut_score = pd.cut(score, 10, labels=qcut_labels)    
    # qcut_count = pd.Series(qcut_score).value_counts().sort_index().values.tolist()

    # TODO: init one time only at the beginning of the app
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=req.scoring_model_name, log_system_metrics=True) as run:
        mlflow.set_tag("Training credit Scoring Model", "Estimator: Logistic Regression")

        logger.info(mlflow.get_tracking_uri())
        logger.info(mlflow.get_artifact_uri())
        
        mlflow.log_params(lr_params)
        mlflow.log_metrics(test_metrics)
        mlflow.log_metrics(train_metrics)

        logger.info('infering MLFlow model I/O signature')
        signature = mlflow.models.infer_signature(X.head(10), params={"predict_method": "score"})

        model_info = mlflow.sklearn.log_model(
            sk_model=scorecard, artifact_path="model", signature=signature,  input_example=X.sample(2)
        )

        dataset = mlflow.data.from_pandas(X)
        mlflow.log_input(dataset, context='train')
        mlflow.log_dict(point_mapping, "point_mapping.json")
        mlflow.log_dict(score_bin_metadata, "score_bin_metadata.json")
        mlflow.log_dict(input_bin_metadata, "input_bin_metadata.json")
        mlflow.log_dict(feat_importance, 'feature_importance.json')

        mlflow_run_id = run.info.run_id
    
    logger.info('testing model inference...')
    model_ = mlflow.sklearn.load_model(model_uri = model_info.model_uri)
    _ = model_.score(X)

    logger.info('recording trained model information to database')
    train_duration_seconds = time.perf_counter() - train_start_at
    train_duration_seconds = round(train_duration_seconds, 0)
    train_duration = round(train_duration_seconds / 60, 2)

    create_model_record = ModelBase(
        id = model_id,
        name = req.scoring_model_name,
        run_id = '',
        status = STATUS_PROCESSING,
        created_at = created_at,
        features = req.features,
        algorithm_id = algorithm_id,
        detail = dict(),
        dataset_table_name = req.table_name,
        dataset_start_date = req.start_date if req.start_date != '' else '1900-01-01',
        dataset_end_date = req.end_date if req.end_date != '' else '2100-01-01',
        updated_at = None,
        deleted_at = None
    )
    _ = crud.create_model(db, create_model_record)

    update_model_record = ModelBase(
        id = model_id,
        name = req.scoring_model_name,
        run_id = mlflow_run_id,
        status = MODEL_STATUS_TRAINED,
        created_at = created_at,
        features = req.features,
        algorithm_id = algorithm_id,
        detail = {
            'risk_definition': req.risk_definition,
            'performance_metrics': {
                'train': train_metrics,
                'test': test_metrics
            },
            'test_ratio': req.test_ratio,
            'test_n_month': req.test_n_month,
            'train_start_date': formatted_train_start,
            'train_end_date': formatted_train_end,
            'test_start_date': formatted_test_start,
            'test_end_date': formatted_test_end
        },
        dataset_table_name = req.table_name,
        dataset_start_date = req.start_date if req.start_date != '' else '1900-01-01',
        dataset_end_date = req.end_date if req.end_date != '' else '2100-01-01',
        updated_at = None,
        deleted_at = None
    )
    _ = crud.create_model(db, update_model_record)

    logger.info(f'training done in {train_duration_seconds // 60} minutes {train_duration_seconds % 60} seconds')
    logger.info(f'algorithm name {algorithm_name}')

    result = ModelResponse(
        id = model_id,
        name = req.scoring_model_name,
        run_id = mlflow_run_id,
        status = MODEL_STATUS_TRAINED,
        created_at = created_at,
        features = req.features,
        algorithm_name = algorithm_name,
        detail = {
            'risk_definition': req.risk_definition,
            'performance_metrics': {
                'train': train_metrics,
                'test': test_metrics
            },
            'test_ratio': req.test_ratio,
            'test_n_month': req.test_n_month,
            'train_start_date': formatted_train_start,
            'train_end_date': formatted_train_end,
            'test_start_date': formatted_test_start,
            'test_end_date': formatted_test_end
        },
        dataset_table_name = req.table_name,
        dataset_start_date = req.start_date if req.start_date != '' else '1900-01-01',
        dataset_end_date = req.end_date if req.end_date != '' else '2100-01-01',
        updated_at = None,
        deleted_at = None
    )
    
    return result


def train_model_async(db: Session, req: TrainRequest, auth_token: str):

    """
    Initiates the model training process asynchronously, returns PID immediately,
    and stores subprocess info for manual tracking.
    """

    
    # Prepare request data

    data = {
        'scoring_model_name': req.scoring_model_name,
        'table_name': req.table_name,
        'start_date': str(req.start_date),
        'end_date': str(req.end_date),
        'features': req.features,
        'algorithm_id': str(req.algorithm_id),
        'selected_risk_definition': req.selected_risk_definition,
        'risk_definition': req.risk_definition,
        'test_ratio': req.test_ratio,
        'test_n_month': req.test_n_month
    }

    json_data = json.dumps(data)

    logger.info("JSON data being sent to the server:")
    logger.info(json_data)

    try:
        curl_command = [
            'curl', '-X', 'POST',
            '-H', 'Content-Type: application/json',

            '-H', f'Authorization: Bearer {auth_token}',

            '-d', json_data,
            '--max-time', str(API_TRAIN_TIMEOUT),  # Timeout for API
            '--fail',  '-L',
            API_TRAIN_ENDPOINT,
            '-w', '%{http_code}'  # Append the HTTP status code to the output
        ]

        # Start the subprocess
        process = subprocess.Popen(
            curl_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Immediately return PID for tracking
        task_id=process.pid
        scoring_model_name=req.scoring_model_name

        train_status = TrainStatusBase(
            task_id=task_id,
            task_status=STATUS_PENDING,
            scoring_model_name=scoring_model_name,
            created_at=datetime.now()
        )

    except Exception as e:
        # Handle unexpected exceptions
        train_status = TrainStatusBase(
            task_id=process.pid,
            task_status= STATUS_FAILED + ' ' + str(e),
            scoring_model_name=scoring_model_name,
            created_at=datetime.now()
        )

    result = crud.create_train_status(db, train_status)

    thread = threading.Thread(target=monitor_process, args=(process, task_id, scoring_model_name))
    thread.daemon = True  # Make sure thread is dead
    thread.start()

    return result

def get_model_simulation(db: Session, req: SimulationRequest):
    try:
        logger.info(f'get model [{req.scoring_model_name}] detail from db')
        model_detail = model_svc.get_model_by_name(db, req.scoring_model_name)
    except NoResultFound:
        logger.exception('model does not exist')
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'model [{req.scoring_model_name}] does not exist')
    
    artifact_uri = f'runs:/{model_detail.run_id}'
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    model_ = mlflow.sklearn.load_model(artifact_uri + '/model')

    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    df = pd.read_sql_table(model_detail.dataset_table_name, engine)
    dataset_detail = dataset_crud.get_dataset_by_name(db, model_detail.dataset_table_name)
    datetime_columns = dataset_detail.datetime_column_names

    if any(col in df.columns for col in datetime_columns):
        df = filter_data_by_datetime(df, datetime_columns, model_detail.dataset_start_date, model_detail.dataset_end_date)

    df = df.dropna(axis=1, how="all")
    df = df.reset_index(drop=True)

    X = df.copy()[model_detail.features].reset_index(drop=True)
    if isinstance(model_detail.detail["risk_definition"], dict):
        y = parse_target_definition(model_detail.detail["risk_definition"], df)
    else:
        y = df[model_detail.detail["risk_definition"]].copy()
        y = np.where(y > 1, 1, 0)
    
    test_ratio = model_detail.detail["test_ratio"]
    test_n_month = model_detail.detail["test_n_month"]

    if test_ratio > 0:
        logger.info(f"Using test_ratio={test_ratio} (prioritized)")
        split_idx = int(len(df) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

    elif test_n_month > 0:
        logger.info(f"test_ratio not valid (test_ratio={test_ratio}). Using test_n_month={test_n_month}")
        threshold = df[datetime_columns[0]].max() - pd.DateOffset(months=test_n_month - 1)
        threshold = threshold.replace(day=1)
        test_mask = df[datetime_columns[0]] >= threshold
        X_train, X_test = X[~test_mask], X[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]

    # Get simulation based on train data
    if req.threshold_train is not None:
        threshold_train_user = round(req.threshold_train)  # IGNORE DATA OUT OF THE INT RANGE (1,99)
        y_score_train = model_.score(X_train)
        y_pred_user = np.where(y_score_train >= threshold_train_user, 0, 1)

        # Get tn,tp,fn,fp,and total cost
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_pred_user).ravel()
        total_cost_train = (tp_train * req.weight_tp_train) + (fp_train * req.weight_fp_train) + (fn_train * req.weight_fn_train) + (tn_train * req.weight_tn_train)

        # Get pnl and threshold data for curve
        profit_loss_curve_train_data = []
        for threshold in range(1, 100):  
            y_pred = np.where(y_score_train >= threshold, 0, 1)  
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_train, y_pred).ravel()
            PnL = ((tp_t * req.weight_tp_train) + (fp_t * req.weight_fp_train) + (fn_t * req.weight_fn_train) + (tn_t * req.weight_tn_train)) * 139676173 
            profit_loss_curve_train_data.append({"x_risk_threshold_train": threshold, "y_pnl_train": PnL})

        # Get pnl of selected threshold
        profit_loss_curve_dict_train = {curve_data["x_risk_threshold_train"]: curve_data["y_pnl_train"] for curve_data in profit_loss_curve_train_data}
        pnl_selected_train = profit_loss_curve_dict_train.get(threshold_train_user)

    
    if req.threshold_test is not None:
        # Get simulation based on test data
        threshold_test_user = round(req.threshold_test) # IGNORE DATA OUT OF THE INT RANGE (1,99)
        y_score_test = model_.score(X_test)
        y_pred_user = np.where(y_score_test >= threshold_test_user, 0, 1)

        # Get tn,tp,fn,fp,and total cost
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_user).ravel()
        total_cost_test = (tp_test * req.weight_tp_test) + (fp_test * req.weight_fp_test) + (fn_test * req.weight_fn_test) + (tn_test * req.weight_tn_test)

        # Get pnl and threshold data for curve
        profit_loss_curve_test_data = []
        for threshold in range(1, 100):  
            y_pred = np.where(y_score_test >= threshold, 0, 1)  
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred).ravel()
            PnL = (((tp_t * req.weight_tp_test) + (fp_t * req.weight_fp_test) + (fn_t * req.weight_fn_test) + (tn_t * req.weight_tn_test))) * 139676173 
            profit_loss_curve_test_data.append({"x_risk_threshold_test": threshold, "y_pnl_test": PnL})

        # Get pnl of selected threshold
        profit_loss_curve_dict_test = {curve_data["x_risk_threshold_test"]: curve_data["y_pnl_test"] for curve_data in profit_loss_curve_test_data}
        pnl_selected_test = profit_loss_curve_dict_test.get(threshold_test_user)

    result = SimulationResponse(
        threshold_train = req.threshold_train,
        tp_train = round(float(tp_train+1e-6),2) if req.weight_tp_train is not None else req.weight_tp_train,
        fp_train = round(float(fp_train+1e-6),2) if req.weight_fp_train is not None else req.weight_fp_train,
        fn_train = round(float(fn_train+1e-6),2) if req.weight_fn_train is not None else req.weight_fn_train,
        tn_train = round(float(tn_train+1e-6),2) if req.weight_tn_train is not None else req.weight_tn_train,
        total_cost_train = round(float(total_cost_train+1e-6),2) if req.weight_fp_train is not None else req.weight_fp_train,
        pnl_selected_train = round(float(pnl_selected_train+1e-6),2) if req.threshold_train is not None else req.threshold_train,
        profit_loss_curve_train = {
        "x_risk_threshold_train": [
            curve_data["x_risk_threshold_train"] for curve_data in profit_loss_curve_train_data
        ] if req.threshold_train is not None else [],
        "y_pnl_train": [
            round(float(curve_data["y_pnl_train"]+1e-6),2) for curve_data in profit_loss_curve_train_data
        ] if req.threshold_train is not None else []
        },
        threshold_test = req.threshold_test,
        tp_test = round(float(tp_test+1e-6),2) if req.weight_tp_test is not None else req.weight_tp_test,
        fp_test = round(float(fp_test+1e-6),2) if req.weight_fp_test is not None else req.weight_fp_test,
        fn_test = round(float(fn_test+1e-6),2) if req.weight_fn_test is not None else req.weight_fn_test,
        tn_test = round(float(tn_test+1e-6),2) if req.weight_tn_test is not None else req.weight_tn_test,
        total_cost_test = round(float(total_cost_test+1e-6),2) if req.weight_fp_test is not None else req.weight_fp_test,
        pnl_selected_test = round(float(pnl_selected_test+1e-6),2) if req.threshold_test is not None else req.threshold_test,
        profit_loss_curve_test = {
        "x_risk_threshold_test": [
            curve_data["x_risk_threshold_test"] for curve_data in profit_loss_curve_test_data
        ] if req.threshold_test is not None else [],
        "y_pnl_test": [
            round(float(curve_data["y_pnl_test"]+1e-6),2) for curve_data in profit_loss_curve_test_data
        ] if req.threshold_test is not None else []
        }
    )
    return result

def monitor_process(process, task_id, scoring_model_name):
    """Monitor subprocess and update DB status based on HTTP status code."""
    
    db = Session
    try:
        while process.poll() is None:
            crud.update_train_status(db, task_id, MODEL_STATUS_TRAINING, 102, STATUS_PROCESSING)
            time.sleep(3)

        stdout, stderr = process.communicate()
        http_status_code = None

        if stdout:
            raw_status_code = stdout[-6:]
            match = re.search(r'\d+', raw_status_code)
            http_status_code = int(match.group()) if match else None

        logger.info(f'status code {http_status_code}')

        if http_status_code and http_status_code > 300:
            status = STATUS_FAILED
            stderr = stderr.split('\n')[-2:-1]
        else:
            status = MODEL_STATUS_TRAINED
            stderr = STATUS_SUCCESS
            http_status_code = 201

        crud.update_train_status(db, task_id, status, http_status_code, stderr)
        crud.update_model_status(db, scoring_model_name, status)
    
    except Exception as e:
        logger.exception(f"monitor_process error: {e}")
    
    finally:
        db.close()  

def update_failed_train_detail(db: Session, req: UpdateFailedTrainRequest):

    model_ = crud.get_model_by_name(db, req.scoring_model_name)
    err_detail = {
        'response_status_code': req.response_status_code,
        'response_detail': req.response_detail
    }

    model_record = ModelBase(
        id = model_.id,
        name = model_.name,
        run_id = STATUS_FAILED,
        status = STATUS_FAILED,
        created_at = model_.created_at,
        features = model_.features,
        algorithm_id = model_.algorithm_id,
        detail = err_detail,
        dataset_table_name = model_.dataset_table_name,
        dataset_start_date = req.start_date if req.start_date != '' else '1900-01-01',
        dataset_end_date = model_.dataset_end_date if model_.dataset_end_date != '' else '2100-01-01'
    )

    result = crud.create_model(db, model_record)

    return result

# feature recommendation
def get_feature_recommendation(db: Session, req: FeatureRecommendationRequest):
    """
    Perform feature selection based on the input parameters and return the selected features.
    """
    feature_columns, target_columns, _, datetime_columns = parse_feature_and_target(db, [], req.table_name)

    logger.info(f'read dataset from table: {req.table_name}')
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    df = pd.read_sql_table(req.table_name, engine)
    if any(col in df.columns for col in datetime_columns):
        df = filter_data_by_datetime(df, datetime_columns, req.start_date, req.end_date)

    df = df.dropna(axis=1, how="all")
    feature_columns = [i for i in feature_columns if i in df.columns]
    X = df.copy()[feature_columns].reset_index(drop=True)
    if isinstance(req.risk_definition, dict):
        y = parse_target_definition(req.risk_definition, df)
    else:
        y = df[req.risk_definition].copy()
        y = np.where(y > 1, 1, 0)

    del df
    gc.collect()

    num_cols = X.select_dtypes(exclude='object').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()
    all_cols = X.columns.tolist()

    logger.info('Starting binning process...')

    # Use the dynamic min_iv and max_iv for feature selection
    selection_criteria = {
        "iv": {"max": req.max_iv, 'min': req.min_iv},
        "quality_score" : {'min' : req.min_qs} # Use min_iv and max_iv dynamically
    }

    binning_process = BinningProcess(variable_names=all_cols,
                                     categorical_variables=cat_cols, 
                                     selection_criteria=selection_criteria)
    binning_process.fit(X, y)

    df_binning_summary = binning_process.summary()
    df_binning_summary['feature_importance'] = df_binning_summary['iv'] / df_binning_summary['iv'].sum()
    logger.info('Binning process done!')
    logger.info(df_binning_summary)

    selected_variables = binning_process.get_support(names=True)
    logger.info(f'selected features: {selected_variables}')
    
    selected_features_list = selected_variables.tolist()
    if len(selected_features_list) > 3:
        recommendation = 'Select the features for training'
    elif len(selected_features_list) > 0 and len(selected_features_list) <= 3:
        recommendation = 'Change the threshold or do not select the features for training'
    else:
        recommendation = 'Change the threshold or select all features for training'

    feature_record = FeatureRecommendationResponse(
        dataset_table_name = req.table_name,
        dataset_start_date = req.start_date,
        dataset_end_date = req.end_date,
        features = selected_features_list,
        recommendation = recommendation)
    
    return feature_record

def get_values_per_column(db: Session, selected_table: str, categorical_features: list, numerical_features: list):
    values_dict = {}

    all_features = categorical_features + numerical_features
    for column_name in all_features:
        # For categorical
        if column_name in categorical_features:
            unique_values_list = dataset_svc.get_dataset_df(db, selected_table, column_name)
            unique_values_list = [value for value in unique_values_list if value != 'nan']
            values_dict[column_name] = unique_values_list
            
        # For numerical
        elif column_name in numerical_features:
            unique_values_list = dataset_svc.get_dataset_df(db, selected_table, column_name)
            min_value = min(unique_values_list)
            max_value = max(unique_values_list)
            
            stats = {
                'min': round(min_value, 2),  
                'max': round(max_value, 2),
            }
            
            values_dict[column_name] = stats  

    return values_dict

def get_features_by_model_name(model_name, db):

    try:
        logger.info(f'get model [{model_name}] detail from db')
        model_detail = model_svc.get_model_by_name(db, model_name)
    except NoResultFound:
        logger.exception('model does not exist')
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'model [{model_name}] does not exist')
    
    if model_detail.status != MODEL_STATUS_DEPLOYED:
        msg = f'model [{model_name}] has not been deployed and now allowed to make prediction'
        logger.warning(msg)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=msg)
    
    # Get feature schema like in db
    selected_table = model_detail.dataset_table_name
    selected_features = model_detail.features
    selected_schema = dataset_svc.check_table(db, selected_table)

    filtered_schema = {key: selected_schema['table_schema'][key] for key in selected_schema['table_schema'].keys() if key in selected_features}
    numerical_features = [key for key, value in filtered_schema.items() if value in ['integer', 'float']]
    categorical_features = [key for key in filtered_schema.keys() if key not in numerical_features]
    unique_values_features = get_values_per_column(db, selected_table, categorical_features, numerical_features)

    FeatureModel = FeatureModelResponse(
        schema_features= filtered_schema,
        unique_values_features=unique_values_features
    )

    return FeatureModel