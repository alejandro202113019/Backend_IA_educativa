# app/services/service_manager.py - ACTUALIZADO CON MODELOS FINE-TUNED
import logging
from typing import Optional
from app.services.enhanced_ai_service import EnhancedAIService
from app.services.nlp_service import NLPService
from app.services.quiz_generator import QuizManager

logger = logging.getLogger(__name__)

class ServiceManager:
    """
    Gestor singleton mejorado para servicios con modelos fine-tuned
    """
    _instance: Optional['ServiceManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'ServiceManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("🚀 Inicializando ServiceManager con modelos fine-tuned")
            self._ai_service: Optional[EnhancedAIService] = None
            self._nlp_service: Optional[NLPService] = None
            self._quiz_manager: Optional[QuizManager] = None
            ServiceManager._initialized = True
        else:
            logger.debug("ServiceManager ya inicializado")
    
    @property
    def ai_service(self) -> EnhancedAIService:
        """Obtiene instancia singleton de EnhancedAIService"""
        if self._ai_service is None:
            logger.info("🤖 Cargando EnhancedAIService con modelos fine-tuned...")
            self._ai_service = EnhancedAIService()
            logger.info("✅ EnhancedAIService cargado y listo")
        return self._ai_service
    
    @property
    def nlp_service(self) -> NLPService:
        """Obtiene instancia singleton de NLPService"""
        if self._nlp_service is None:
            logger.info("🔤 Cargando NLPService...")
            self._nlp_service = NLPService()
            logger.info("✅ NLPService cargado y listo")
        return self._nlp_service
    
    @property
    def quiz_manager(self) -> QuizManager:
        """Obtiene instancia singleton de QuizManager"""
        if self._quiz_manager is None:
            logger.info("📝 Cargando QuizManager...")
            self._quiz_manager = QuizManager()
            logger.info("✅ QuizManager cargado y listo")
        return self._quiz_manager
    
    def preload_all_services(self):
        """Pre-cargar todos los servicios mejorados"""
        logger.info("⚡ Pre-cargando todos los servicios mejorados...")
        _ = self.ai_service
        _ = self.nlp_service  
        _ = self.quiz_manager
        logger.info("🎯 Todos los servicios mejorados pre-cargados")
    
    def get_status(self) -> dict:
        """Obtiene el estado completo de todos los servicios"""
        base_status = {
            "ai_service_loaded": self._ai_service is not None,
            "nlp_service_loaded": self._nlp_service is not None,
            "quiz_manager_loaded": self._quiz_manager is not None,
            "enhanced_manager_initialized": self._initialized
        }
        
        # Agregar estado detallado de modelos si AI service está cargado
        if self._ai_service:
            base_status["model_details"] = self._ai_service.get_model_status()
        
        return base_status

# Instancia global mejorada
service_manager = ServiceManager()
