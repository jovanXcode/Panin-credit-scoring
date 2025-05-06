from pydantic import BaseModel, Field

from src.config.constant import STATUS_SUCCESS

class DeploymentRequest(BaseModel):
    deployed_model_name: str = Field(examples=["model-trial-12"])

class DeploymentResponse(BaseModel):
    deployed_model_name: str = Field(examples=["model-trial-12"])
    deployed_model_run_id: str = Field(examples=["918495eaf2dd447497f368bb44b78c78"])
    deploy_status: str = Field(examples=[STATUS_SUCCESS])

class UndeploymentResponse(BaseModel):
    undeployed_model_name: str = Field(examples=["model-trial-12"])
    undeployed_model_run_id: str = Field(examples=["918495eaf2dd447497f368bb44b78c78"])
    undeploy_status: str = Field(examples=[STATUS_SUCCESS])
