# app/core/config.py CORREGIDO
import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "IA Educativa Backend"
    VERSION: str = "1.0.0"
    
    # Security
    SECRET_KEY: str = "fallback-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # OpenAI
    OPENAI_API_KEY: str = ""
    
    # File Upload
    MAX_FILE_SIZE: int = 10485760  # 10MB
    ALLOWED_EXTENSIONS: str = "pdf,txt,docx"  # ← CAMBIO: str en lugar de List[str]
    UPLOAD_FOLDER: str = "uploads"
    TEMP_FOLDER: str = "temp"
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Método para obtener extensiones como lista
    def get_allowed_extensions(self) -> List[str]:
        return self.ALLOWED_EXTENSIONS.split(",")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

