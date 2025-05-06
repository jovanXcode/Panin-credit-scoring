from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.routers.v1 import models as models_router
from src.routers.v1 import features as features_router
from src.routers.v1 import datasets as datasets_router
from utils.auth import get_current_user

# TODO: replace database migration method with alembic
# scoring_history.Base.metadata.create_all(bind=engine)
# scoring_schedule.Base.metadata.create_all(bind=engine)
# model.Base.metadata.create_all(bind=engine)
# dataset.Base.metadata.create_all(bind=engine)

# TODO: init mlflow client 1 time on app startup
# @app.on_startup

app = FastAPI(
    title="AI-CREDIT-SCORING-API",
    description="welcome to AI Credit Scoring service documentation",
    summary="hola",
    version="0.0.2",    
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models_router.router, prefix='/api', dependencies=[Depends(get_current_user)])
app.include_router(features_router.router, prefix='/api', dependencies=[Depends(get_current_user)])
app.include_router(datasets_router.router, prefix='/api', dependencies=[Depends(get_current_user)])

@app.get("/test-auth", dependencies=[Depends(get_current_user)])
async def secure_data():
    return {"message": "You are authorized"}

@app.get("/health")
async def health():
    return {"message": "OK"}