import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")
    GEMINI_MODEL: str = Field(default="gemini-3.5-flash", env="GEMINI_MODEL")
    MLFLOW_TRACKING_URI: str = Field(default="sqlite:///backend/data/mlflow.db", env="MLFLOW_TRACKING_URI")
    CHROMA_DB_PATH: str = Field(default="backend/data/chroma", env="CHROMA_DB_PATH")
    SEC_USER_AGENT: str = Field(default="FinSightAI-DevUser student_dev@finsightai.local", env="SEC_USER_AGENT")
    HOST: str = Field(default="127.0.0.1", env="HOST")
    PORT: int = Field(default=8000, env="PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
