import uuid
from sqlalchemy import Uuid
from sqlalchemy import Column
from sqlalchemy import String, Date, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from src.utils.database import Base

class Model(Base):
    __tablename__ =  "models"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    name = Column(String, primary_key=True)
    run_id = Column(String)
    status = Column(String)
    dataset_table_name = Column(String)
    dataset_start_date = Column(Date)
    dataset_end_date = Column(Date)
    features = Column(ARRAY(String))
    algorithm_id = Column(Uuid, ForeignKey('algorithms.id'), primary_key=False, nullable=False)
    detail = Column(JSONB)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    deleted_at = Column(DateTime)
    
    algorithm = relationship(
        "Algorithm",
        primaryjoin="Model.algorithm_id == Algorithm.id",
        viewonly=True,  # Ensures it's read-only
        uselist=False
    )

    @hybrid_property
    def algorithm_name(self):
        return self.algorithm.name if self.algorithm else ''

class ModelTrainingStatus(Base):
    __tablename__ = "model_training_status"
    
    task_id = Column(Integer, primary_key=True)
    task_status = Column(String)
    status_code = Column(Integer)
    status_code_detail = Column(String)
    scoring_model_name = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    deleted_at = Column(DateTime)

class Algorithm(Base):
    __tablename__ =  "algorithms"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(String)