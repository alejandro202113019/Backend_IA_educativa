# app/main.py - ACTUALIZADO
import logging
import uvicorn
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.routes import upload, summary, quiz
from app.models.response_models import APIResponse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestión del ciclo de vida de la aplicación
    """
    # Startup
    logger.info("Iniciando aplicación IA Educativa Backend con modelos gratuitos")
    
    # Crear directorios necesarios
    import os
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(settings.TEMP_FOLDER, exist_ok=True)
    os.makedirs(settings.AI_MODEL_CACHE_DIR, exist_ok=True)
    
    # YA NO VERIFICAMOS OPENAI_API_KEY
    # Verificar que los modelos se puedan cargar
    try:
        from app.services.ai_service import AIService
        ai_service = AIService()
        logger.info("✅ Modelos de IA cargados correctamente")
    except Exception as e:
        logger.warning(f"⚠️ Error cargando modelos de IA: {e}")
        logger.info("La aplicación funcionará con capacidades limitadas")
    
    yield
    
    # Shutdown
    logger.info("Cerrando aplicación...")

# Crear aplicación FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API para procesamiento de contenido educativo con IA gratuita (BART, T5, Stable Diffusion)",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(
    upload.router,
    prefix=f"{settings.API_V1_STR}/upload",
    tags=["upload"]
)

app.include_router(
    summary.router,
    prefix=f"{settings.API_V1_STR}/summary",
    tags=["summary"]
)

app.include_router(
    quiz.router,
    prefix=f"{settings.API_V1_STR}/quiz",
    tags=["quiz"]
)

# Manejadores de excepciones globales
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=APIResponse(
            success=False,
            message=exc.detail,
            error=f"HTTP {exc.status_code}"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Error no manejado: {exc}")
    return JSONResponse(
        status_code=500,
        content=APIResponse(
            success=False,
            message="Error interno del servidor",
            error="INTERNAL_SERVER_ERROR"
        ).dict()
    )

# Endpoints básicos
@app.get("/", response_model=APIResponse)
async def root():
    """
    Endpoint raíz - información de la API
    """
    return APIResponse(
        success=True,
        message="IA Educativa Backend API con modelos gratuitos está funcionando",
        data={
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "ai_models": "BART (resúmenes), T5 (quiz), Transformers (análisis)",
            "gpu_available": "cuda" if settings.AI_USE_GPU else "cpu",
            "endpoints": {
                "upload": f"{settings.API_V1_STR}/upload",
                "summary": f"{settings.API_V1_STR}/summary", 
                "quiz": f"{settings.API_V1_STR}/quiz"
            }
        }
    )

@app.get("/health", response_model=APIResponse)
async def health_check():
    """
    Endpoint de salud del sistema
    """
    # Verificar estado de los modelos
    try:
        import torch
        gpu_status = "disponible" if torch.cuda.is_available() else "no disponible"
    except:
        gpu_status = "no detectada"
    
    return APIResponse(
        success=True,
        message="Sistema saludable - Modelos gratuitos funcionando",
        data={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": settings.ENVIRONMENT,
            "gpu": gpu_status,
            "models": "BART + T5 + Transformers"
        }
    )

# Punto de entrada para desarrollo
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )