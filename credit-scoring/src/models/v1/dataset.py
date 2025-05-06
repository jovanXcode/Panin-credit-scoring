import uuid
from datetime import datetime
from sqlalchemy import Uuid
from sqlalchemy import Column, func
from sqlalchemy import Integer, String, Date, DateTime
from sqlalchemy.dialects.postgresql import ARRAY

from src.utils.database import Base

class Dataset(Base):
    __tablename__ =  "datasets"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    table_name = Column(String)
    target_column_names = Column(ARRAY(String))
    identity_column_names = Column(ARRAY(String))
    datetime_column_names = Column(ARRAY(String))
    description = Column(String)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime)
    deleted_at = Column(DateTime)