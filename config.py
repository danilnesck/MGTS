from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Добавили +pg8000 после postgresql
    # Убедись, что тут psycopg2 и порт 5433
    # Убедись, что тут psycopg2 и порт 5433
    DATABASE_URL: str = "postgresql+psycopg2://mts_user:mts_password@127.0.0.1:5433/mts_hackathon"
    CONFIDENCE_THRESHOLD: float = 0.5 

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()