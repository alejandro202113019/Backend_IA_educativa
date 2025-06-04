# app/services/service_manager.py - NUEVO ARCHIVO
import logging
from typing import Optional
from app.services.ai_service import AIService
from app.services.nlp_service import NLPService
from app.services.quiz_generator import QuizManager

logger = logging.getLogger(__name__)

class ServiceManager:
    """
    Gestor singleton para todos los servicios de IA
    Garantiza que los modelos se carguen una sola vez
    """
    _instance: Optional['ServiceManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'ServiceManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("🚀 Inicializando ServiceManager (primera vez)")
            self._ai_service: Optional[AIService] = None
            self._nlp_service: Optional[NLPService] = None
            self._quiz_manager: Optional[QuizManager] = None
            ServiceManager._initialized = True
        else:
            logger.debug("ServiceManager ya inicializado (reutilizando instancia)")
    
    @property
    def ai_service(self) -> AIService:
        """Obtiene instancia singleton de AIService"""
        if self._ai_service is None:
            logger.info("🤖 Cargando AIService por primera vez...")
            self._ai_service = AIService()
            logger.info("✅ AIService cargado y listo")
        return self._ai_service
    
    @property
    def nlp_service(self) -> NLPService:
        """Obtiene instancia singleton de NLPService"""
        if self._nlp_service is None:
            logger.info("🔤 Cargando NLPService por primera vez...")
            self._nlp_service = NLPService()
            logger.info("✅ NLPService cargado y listo")
        return self._nlp_service
    
    @property
    def quiz_manager(self) -> QuizManager:
        """Obtiene instancia singleton de QuizManager"""
        if self._quiz_manager is None:
            logger.info("📝 Cargando QuizManager por primera vez...")
            self._quiz_manager = QuizManager()
            logger.info("✅ QuizManager cargado y listo")
        return self._quiz_manager
    
    def preload_all_services(self):
        """Pre-cargar todos los servicios (útil en startup)"""
        logger.info("⚡ Pre-cargando todos los servicios...")
        _ = self.ai_service
        _ = self.nlp_service  
        _ = self.quiz_manager
        logger.info("🎯 Todos los servicios pre-cargados")
    
    def get_status(self) -> dict:
        """Obtiene el estado de todos los servicios"""
        return {
            "ai_service_loaded": self._ai_service is not None,
            "nlp_service_loaded": self._nlp_service is not None,
            "quiz_manager_loaded": self._quiz_manager is not None,
            "singleton_initialized": self._initialized
        }

# Instancia global singleton
service_manager = ServiceManager()