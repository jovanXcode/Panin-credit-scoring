from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.config.constant import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_AUTH_DATABASE_URI

engine = create_engine(SQLALCHEMY_DATABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

engine_auth = create_engine(SQLALCHEMY_AUTH_DATABASE_URI)
SessionAuthLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_auth)

Base = declarative_base()
BaseAuth = declarative_base()
# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_auth_db():
    db = SessionAuthLocal()
    try:
        yield db
    finally:
        db.close()