# app/core/config.py - VERSIÓN SIN OPENAI
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
    
    # YA NO NECESITAMOS OPENAI_API_KEY
    # OPENAI_API_KEY: str = ""  # <-- ELIMINAR ESTA LÍNEA
    
    # Configuración de modelos de IA
    AI_MODEL_CACHE_DIR: str = "model_cache"
    AI_USE_GPU: bool = True  # Usar GPU si está disponible
    AI_MODEL_SIZE: str = "base"  # base, small, large
    
    # File Upload
    MAX_FILE_SIZE: int = 10485760  # 10MB
    ALLOWED_EXTENSIONS: str = "pdf,txt,docx"
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