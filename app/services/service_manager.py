# app/services/service_manager.py - ACTUALIZADO CON SERVICIOS MEJORADOS
import logging
from typing import Optional
from app.services.improved_ai_service import ImprovedAIService
from app.services.nlp_service import UniversalNLPService
from app.services.quiz_generator import QuizManager

logger = logging.getLogger(__name__)

class ServiceManager:
    """
    Gestor singleton mejorado con servicios optimizados para cualquier tipo de texto
    """
    _instance: Optional['ServiceManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'ServiceManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("🚀 Inicializando ServiceManager con servicios mejorados universales")
            self._ai_service: Optional[ImprovedAIService] = None
            self._nlp_service: Optional[UniversalNLPService] = None
            self._quiz_manager: Optional[QuizManager] = None
            ServiceManager._initialized = True
        else:
            logger.debug("ServiceManager ya inicializado")
    
    @property
    def ai_service(self) -> ImprovedAIService:
        """Obtiene instancia singleton de ImprovedAIService"""
        if self._ai_service is None:
            logger.info("🤖 Cargando ImprovedAIService con prompts especializados...")
            self._ai_service = ImprovedAIService()
            logger.info("✅ ImprovedAIService cargado y optimizado")
        return self._ai_service
    
    @property
    def nlp_service(self) -> UniversalNLPService:
        """Obtiene instancia singleton de UniversalNLPService"""
        if self._nlp_service is None:
            logger.info("🔤 Cargando UniversalNLPService...")
            self._nlp_service = UniversalNLPService()
            logger.info("✅ UniversalNLPService cargado y optimizado")
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
        logger.info("⚡ Pre-cargando todos los servicios mejorados universales...")
        _ = self.ai_service
        _ = self.nlp_service  
        _ = self.quiz_manager
        logger.info("🎯 Todos los servicios mejorados pre-cargados exitosamente")
    
    def get_status(self) -> dict:
        """Obtiene el estado completo de todos los servicios mejorados"""
        base_status = {
            "ai_service_loaded": self._ai_service is not None,
            "nlp_service_loaded": self._nlp_service is not None,
            "quiz_manager_loaded": self._quiz_manager is not None,
            "service_manager_version": "improved_universal_v2.0",
            "features": [
                "Universal domain detection",
                "Specialized prompts",
                "Advanced post-processing",
                "Contextual question generation",
                "Personalized feedback"
            ]
        }
        
        # Agregar estado detallado si los servicios están cargados
        if self._ai_service:
            base_status["ai_service_details"] = {
                "type": "ImprovedAIService",
                "device": getattr(self._ai_service, 'device', 'unknown'),
                "models_loaded": True,
                "specialized_prompts": True,
                "domain_detection": True
            }
        
        if self._nlp_service:
            base_status["nlp_service_details"] = {
                "type": "UniversalNLPService", 
                "spacy_model": "loaded" if getattr(self._nlp_service, 'nlp', None) else "fallback",
                "domain_patterns": len(getattr(self._nlp_service, 'domain_patterns', {})),
                "universal_processing": True
            }
        
        return base_status

# Instancia global mejorada
service_manager = ServiceManager()