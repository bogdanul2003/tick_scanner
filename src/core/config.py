"""Application configuration management."""
import os
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database settings
    db_name: str = "ticks"
    db_user: str = "postgres"
    db_password: str = "postgres"
    db_host: str = "localhost"
    db_port: int = 5432
    db_pool_min: int = 1
    db_pool_max: int = 15
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # CORS settings
    cors_origins: List[str] = ["http://localhost:5173"]
    
    # Chart generation settings
    charts_base_dir: str = "../generated_charts"
    
    # Forecast settings
    forecast_max_workers: int = 4
    forecast_days_past: int = 100
    forecast_days_ahead: int = 5
    
    # MACD settings
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    
    # Pattern detection settings
    pattern_days_default: int = 120
    
    class Config:
        env_prefix = "TICK_SCANNER_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def db_params(self) -> dict:
        """Get database connection parameters."""
        return {
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
            "host": self.db_host,
            "port": self.db_port,
        }
    
    @property
    def database_url(self) -> str:
        """Get database URL for connection string."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses lru_cache to avoid re-reading environment on every access.
    """
    return Settings()


# Global settings instance for easy import
settings = get_settings()
