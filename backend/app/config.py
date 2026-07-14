from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash"
    MLFLOW_TRACKING_URI: str = "sqlite:///backend/data/mlflow.db"
    CHROMA_DB_PATH: str = "backend/data/chroma"
    SEC_USER_AGENT: str = "FinLensAI-DevUser student_dev@finlensai.local"
    HOST: str = "127.0.0.1"
    PORT: int = 8000

settings = Settings()
