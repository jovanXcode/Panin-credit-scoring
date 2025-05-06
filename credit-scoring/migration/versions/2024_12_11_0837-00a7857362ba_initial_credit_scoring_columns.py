"""initial credit scoring columns

Revision ID: 00a7857362ba
Revises: 
Create Date: 2024-12-11 08:37:13.615662

"""
import os
import uuid
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

from src.config.constant import DUMMY_DATASET_TABLE_NAME, ALGORITHM_DATA

# revision identifiers, used by Alembic.
revision: str = '00a7857362ba'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    ENV = os.environ.get("ENV")
    if ENV == 'DEV':
        _ = op.create_table(
            DUMMY_DATASET_TABLE_NAME,
            sa.Column('customer_id', sa.String),
            sa.Column('loan_id', sa.String),
            sa.Column('purpose', sa.String),
            sa.Column('tenure', sa.Integer),
            sa.Column('interest_rate', sa.Float),
            sa.Column('debt_to_income', sa.Float),
            sa.Column('vehicle_type', sa.String),
            sa.Column('education', sa.String),
            sa.Column('occupation', sa.String),
            sa.Column('job_title', sa.String),
            sa.Column('job_industry', sa.String),
            sa.Column('income_type', sa.String),
            sa.Column('criminal_record', sa.String),
            sa.Column('marital_status', sa.String),
            sa.Column('age', sa.Integer),
            sa.Column('income_class', sa.Integer),
            sa.Column('num_work_year', sa.Integer),
            sa.Column('num_dependent', sa.Integer),
            sa.Column('house_type', sa.String),
            sa.Column('length_of_stay', sa.Integer),
            sa.Column('residency_status', sa.String),
            sa.Column('house_ownership', sa.String),
            sa.Column('earning_members', sa.Integer),
            sa.Column('vehicle_own', sa.Integer),
            sa.Column('is_phone_number_owner', sa.String),
            sa.Column('dpd_m1', sa.Integer),
            sa.Column('col_m1', sa.Integer),
            sa.Column('utilization_per_credit_account', sa.Float),
            sa.Column('utilization_per_credit_amount', sa.Float),
            sa.Column('latest_per_total_payment_m1', sa.Float),
            sa.Column('age_oldest_account', sa.Integer),
            sa.Column('datetime', sa.DateTime)
        )

        op.execute(f"COPY {DUMMY_DATASET_TABLE_NAME} FROM '/tmp/dataset.csv' DELIMITER ',' CSV HEADER;", execution_options=None)

    _ = op.create_table(
        'datasets',
        sa.Column('id', sa.Uuid, primary_key=True),
        sa.Column('table_name', sa.String, unique=True, nullable=False),
        sa.Column('target_column_names', sa.dialects.postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('identity_column_names', sa.dialects.postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('datetime_column_names', sa.dialects.postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('description', sa.String, unique=False, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=True),
        sa.Column('deleted_at', sa.DateTime, nullable=True),
    )

    _ = op.create_table(
        'scoring_schedule',
        sa.Column('schedule_id', sa.Uuid, primary_key=True),
        sa.Column('scoring_model_name', sa.String, unique=True, nullable=False),
        sa.Column('input_table_name', sa.String, unique=False, nullable=False),
        sa.Column('input_start_date', sa.Date, unique=False, nullable=True),
        sa.Column('input_end_date', sa.Date, unique=False, nullable=True),
        sa.Column('taks_key', sa.String, unique=False, nullable=False),
        sa.Column('cron_format', sa.String, unique=False, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=True),
        sa.Column('deleted_at', sa.DateTime, nullable=True),
    )

    _ = op.create_table(
        'scoring_history',
        sa.Column('predict_id', sa.Uuid, primary_key=True),
        sa.Column('app_id', sa.String, unique=False, nullable=False),
        sa.Column('scoring_model_name', sa.String, unique=False, nullable=False),
        sa.Column('app_detail', sa.dialects.postgresql.JSONB, unique=False, nullable=False),
        sa.Column('output_table_name', sa.String, unique=False, nullable=True),
        sa.Column('input', sa.dialects.postgresql.ARRAY(sa.dialects.postgresql.JSONB), nullable=True),
        sa.Column('result', sa.dialects.postgresql.ARRAY(sa.dialects.postgresql.JSONB), nullable=True),
        sa.Column('detail', sa.dialects.postgresql.JSONB, unique=False, nullable=False),
        sa.Column('predicted_at', sa.DateTime, nullable=False)
    )

    _ = op.create_table(
        'monitor_history',
        sa.Column('monitor_id', sa.Uuid, primary_key=True),
        sa.Column('monitored_at', sa.DateTime, nullable=False),
        sa.Column('scoring_model_name', sa.String, unique=False, nullable=False),
        sa.Column('psi', sa.Float),
        sa.Column('kl_mean', sa.Float),
        sa.Column('kl_median', sa.Float),
        sa.Column('input_drift_detail', sa.dialects.postgresql.JSONB, unique=False, nullable=False),
        sa.Column('num_account', sa.Integer),
        sa.Column('min_score', sa.Float),
        sa.Column('max_score', sa.Float),
        sa.Column('avg_score', sa.Float),
        sa.Column('median_score', sa.Float),
        sa.Column('std_score', sa.Float)
    )

    _ = op.create_table(
        'model_training_status',
        sa.Column('task_id', sa.Integer, primary_key=True),
        sa.Column('task_status', sa.String),
        sa.Column('status_code', sa.Integer),
        sa.Column('status_code_detail', sa.String),
        sa.Column('scoring_model_name', sa.String, unique=False, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=True),
        sa.Column('deleted_at', sa.DateTime, nullable=True)
    )

    _ = op.create_table(
        'scoring_result',
        sa.Column('predict_id', sa.Uuid,),
        sa.Column('app_id', sa.String, unique=False, nullable=False),
        sa.Column('app_detail', sa.dialects.postgresql.JSONB, unique=False, nullable=False),
        sa.Column('input_detail', sa.dialects.postgresql.JSONB),
        sa.Column('output_detail', sa.dialects.postgresql.JSONB),
        sa.Column('score_breakdown', sa.dialects.postgresql.JSONB),
        sa.Column('total_score', sa.Float),
        sa.Column('scoring_model_name', sa.String),
        sa.Column('scoring_model_run_id', sa.String),
        sa.Column('predicted_at', sa.DateTime, nullable=False)
    )

    algorithms_table = op.create_table(
        'algorithms',
        sa.Column('id', sa.Uuid, primary_key=True),
        sa.Column('name', sa.String),
        sa.Column('description', sa.String)

    )
    op.bulk_insert(algorithms_table, ALGORITHM_DATA)

    _ = op.create_table(
        'models',
        sa.Column('id', sa.Uuid, primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String, unique=True, nullable=False),
        sa.Column('run_id', sa.String, nullable=False),
        sa.Column('status', sa.String, nullable=False),
        sa.Column('dataset_table_name', sa.String, nullable=False),
        sa.Column('dataset_start_date', sa.Date, nullable=True),
        sa.Column('dataset_end_date', sa.Date, nullable=True),
        sa.Column('features', sa.dialects.postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('algorithm_id', sa.Uuid, sa.ForeignKey('algorithms.id'), primary_key=False, nullable=False),
        sa.Column('detail', sa.dialects.postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=True),
        sa.Column('updated_at', sa.DateTime, nullable=True, server_onupdate=sa.func.now()),
        sa.Column('deleted_at', sa.DateTime, nullable=True),
    )

def downgrade() -> None:

    # op.drop_table(DUMMY_DATASET_TABLE_NAME)
    op.drop_table('datasets')
    op.drop_table('models')
    op.drop_table('scoring_schedule')
    op.drop_table('scoring_history')
    op.drop_table('monitor_history')
    op.drop_table('model_training_status')
    op.drop_table('scoring_result')
    op.drop_table('algorithms')