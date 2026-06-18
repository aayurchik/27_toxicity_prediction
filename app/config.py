import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env файл
load_dotenv()


class Settings:
    """Настройки приложения"""
    
    # ===== База данных =====
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    
    # ===== Модель =====
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models_data/rb_model.pt")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "models_data/scaler.pkl")
    INPUT_SIZE: int = int(os.getenv("INPUT_SIZE", "78"))
    OUTPUT_SIZE: int = int(os.getenv("OUTPUT_SIZE", "13"))
    
    # ===== Безопасность =====
    ADMIN_TOKEN: str = os.getenv("ADMIN_TOKEN", "secretoken")
    
    # ===== FastAPI =====
    API_TITLE: str = "FastAPI Toxicity Prediction"
    API_DESCRIPTION: str = "Prediction of toxicity of chemical compounds based on their physicochemical properties"
    API_VERSION: str = "1.0.0"


settings = Settings()