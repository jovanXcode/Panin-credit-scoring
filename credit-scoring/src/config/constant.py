import os
import uuid

# from dotenv import load_dotenv
# from pathlib import Path

# dotenv_path = Path('/home/soncom/credit-ai-module/credit-scoring/.env')
# load_dotenv(dotenv_path=dotenv_path)

AI_SERVICE_HOST = os.getenv('AI_SERVICE_HOST') 
AI_SERVICE_PORT = os.getenv('API_PORT')

AI_TRAIN_ENDPOINT = os.getenv('AI_TRAIN_ENDPOINT') 
AI_PREDICT_ENDPOINT = os.getenv('AI_PREDICT_ENDPOINT') 
AI_FAILED_TRAIN_ENDPOINT = os.getenv('AI_FAILED_TRAIN_ENDPOINT') 

API_TRAIN_ENDPOINT = f'http://{AI_SERVICE_HOST}:{AI_SERVICE_PORT}/{AI_TRAIN_ENDPOINT}'
API_PREDICT_ENDPOINT = f'http://{AI_SERVICE_HOST}:{AI_SERVICE_PORT}/{AI_PREDICT_ENDPOINT}'
API_FAILED_TRAIN_ENDPOINT = f'http://{AI_SERVICE_HOST}:{AI_SERVICE_PORT}/{AI_FAILED_TRAIN_ENDPOINT}'

API_TRAIN_TIMEOUT = int(os.getenv('AI_TRAIN_TIMEOUT'))
API_PREDICT_TIMEOUT = int(os.getenv('AI_PREDICT_TIMEOUT'))
AI_FAILED_TRAIN_TIMEOUT = int(os.getenv('AI_FAILED_TRAIN_TIMEOUT'))

DB_POSTGRES_HOST = os.getenv("DB_POSTGRES_HOST")
DB_POSTGRES_USER = os.getenv("DB_POSTGRES_USER")
DB_POSTGRES_PASSWORD=os.getenv("DB_POSTGRES_PASSWORD")
DB_POSTGRES_PORT=os.getenv("DB_POSTGRES_PORT")
DB_POSTGRES_NAME=os.getenv("DB_POSTGRES_NAME")

DB_AUTH_HOST = os.getenv("AUTH_DB_HOST")  # Authentication DB
DB_AUTH_USER = os.getenv("AUTH_DB_USER")
DB_AUTH_PASSWORD = os.getenv("AUTH_DB_PASSWORD")
DB_AUTH_PORT = os.getenv("AUTH_DB_PORT")
DB_AUTH_NAME = os.getenv("AUTH_DB_NAME")

SQLALCHEMY_DATABASE_URI = f'postgresql://{DB_POSTGRES_USER}:{DB_POSTGRES_PASSWORD}@{DB_POSTGRES_HOST}:{DB_POSTGRES_PORT}/{DB_POSTGRES_NAME}'
SQLALCHEMY_AUTH_DATABASE_URI = f"postgresql://{DB_AUTH_USER}:{DB_AUTH_PASSWORD}@{DB_AUTH_HOST}:{DB_AUTH_PORT}/{DB_AUTH_NAME}"

MLFLOW_HOST = "mlflow"
MLFLOW_PORT = os.getenv("MLFLOW_PORT")
MLFLOW_EXPERIMENT_NAME = "credit_scoring"
MLFLOW_TRACKING_URI = f'http://{MLFLOW_HOST}:{MLFLOW_PORT}'

DUMMY_DATASET_TABLE_NAME = os.getenv("DUMMY_DATASET_TABLE_NAME")

MODEL_STATUS_TRAINING = 'TRAINING'
MODEL_STATUS_TRAINED = 'TRAINED'
MODEL_STATUS_DEPLOYED = 'DEPLOYED'
MODEL_STATUS_CHALLENGER = 'CHALLENGER'

REDIS_URI = os.getenv('REDIS_URI')

STATUS_SUCCESS = 'SUCCESS'
STATUS_PENDING = 'PENDING'
STATUS_FAILED = 'FAILED'
STATUS_PROCESSING = 'PROCESSING'

MIN_RISK_SCORE = 0
MAX_RISK_SCORE = 100

TYPE_CATEGORICAL = 'CATEGORICAL'
TYPE_NUMERICAL = 'NUMERICAL'


ALGORITHM_DATA = [
        {"id": str(uuid.uuid4()), "name": "LR-ScoreCard", "description": "A model uses logistic regression to predict the likelihood of an event; its advantage is that it is easy to understand and fast in processing data, while its drawback is that it cannot capture complex patterns and may be affected by highly correlated data."},
        {"id": str(uuid.uuid4()), "name": "Elasticnet-ScoreCard", "description": "A model combines two methods to make predictions more accurate and avoid errors; its advantage is that it is effective at selecting important features and works well with complicated data, while its drawback is that it requires careful tuning and can be hard to understand."}
        # {"id": str(uuid.uuid4()), "name": "LRCV-ScoreCard", "description": "A logistic regression model that uses a technique to test its accuracy; its advantage is that it gives more reliable results, but its drawback is that the testing process can take time and be complex."},
        # {"id": str(uuid.uuid4()), "name": "ElasticnetCV-ScoreCard", "description": "A model uses Elastic Net with testing to find the best way to make predictions; its advantage is that it provides highly accurate results and automatically selects important features, but it can be hard to understand and requires deep knowledge."}
    ]

# Algorithm name
SCORECARD_LR = 'LR-ScoreCard'
SCORECARD_LRCV = 'LRCV-ScoreCard'
SCORECARD_ELASTICNETCV = 'ElasticnetCV-ScoreCard'
SCORECARD_ELASTICNET = 'Elasticnet-ScoreCard'

DECISION_ACCEPT = 'ACCEPT'
DECISION_CHECK = 'CHECK'
DECISION_REJECT = 'REJECT'