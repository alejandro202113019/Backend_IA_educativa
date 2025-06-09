# app/services/service_manager.py - CORREGIDO PARA USAR GENERADOR SIMPLE
import logging
import re
from typing import Optional, List
from app.services.simple_ai_service import SimpleAIServiceWithContextualQuiz
from app.services.nlp_service import UniversalNLPService
from app.services.quiz_generator import QuizManager

logger = logging.getLogger(__name__)

class ServiceManager:
    """
    Gestor singleton con generador simple y efectivo de quiz contextual
    """
    _instance: Optional['ServiceManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'ServiceManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("🚀 Inicializando ServiceManager con Quiz Contextual Simple y Efectivo")
            self._ai_service: Optional[SimpleAIServiceWithContextualQuiz] = None
            self._nlp_service: Optional[UniversalNLPService] = None
            self._quiz_manager: Optional[QuizManager] = None
            ServiceManager._initialized = True
        else:
            logger.debug("ServiceManager ya inicializado")
    
    @property
    def ai_service(self) -> SimpleAIServiceWithContextualQuiz:
        """Obtiene instancia singleton de SimpleAIServiceWithContextualQuiz"""
        if self._ai_service is None:
            logger.info("🤖 Cargando SimpleAIServiceWithContextualQuiz...")
            self._ai_service = SimpleAIServiceWithContextualQuiz()
            logger.info("✅ Servicio de IA simple con quiz contextual cargado")
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
        """Pre-cargar todos los servicios simples"""
        logger.info("⚡ Pre-cargando servicios simples con generador contextual...")
        _ = self.ai_service
        _ = self.nlp_service  
        _ = self.quiz_manager
        logger.info("🎯 Servicios simples con quiz contextual pre-cargados exitosamente")
    
    def get_status(self) -> dict:
        """Obtiene el estado completo del sistema simplificado"""
        base_status = {
            "ai_service_loaded": self._ai_service is not None,
            "nlp_service_loaded": self._nlp_service is not None,
            "quiz_manager_loaded": self._quiz_manager is not None,
            "service_manager_version": "simple_contextual_v1.0",
            "features": [
                "Simple contextual quiz generation",
                "Direct content extraction", 
                "Fact-based questions",
                "WWII specific patterns",
                "Emergency fallback system",
                "Text-specific analysis"
            ]
        }
        
        if self._ai_service:
            base_status["ai_service_details"] = {
                "type": "SimpleAIServiceWithContextualQuiz",
                "contextual_generator": True,
                "wwii_patterns": True,
                "fact_extraction": True,
                "emergency_fallback": True,
                "supported_domains": ["historia_wwii", "general"]
            }
        
        if self._nlp_service:
            base_status["nlp_service_details"] = {
                "type": "UniversalNLPService", 
                "domain_detection": True,
                "concept_extraction": True
            }
        
        return base_status
    
    def test_quiz_with_wwii_content(self) -> dict:
        """
        Prueba específica con contenido de la Segunda Guerra Mundial
        """
        wwii_test_text = """La Segunda Guerra Mundial comenzó el 1 de septiembre de 1939 cuando Alemania Nazi invadió Polonia. 
        Adolf Hitler había planeado esta invasión como parte de su estrategia para expandir el territorio alemán.
        
        Francia e Inglaterra declararon la guerra a Alemania el 3 de septiembre de 1939, dando inicio oficial 
        al conflicto que se convertiría en la guerra más devastadora de la historia.
        
        La estrategia militar alemana se basaba en el Blitzkrieg (guerra relámpago), que combinaba el uso 
        coordinado de tanques, aviación de combate y tropas móviles para lograr victorias rápidas y decisivas.
        
        En 1941, la guerra se expandió globalmente cuando Alemania lanzó la Operación Barbarroja contra 
        la Unión Soviética y Japón atacó la base naval estadounidense de Pearl Harbor.
        
        El punto de inflexión del conflicto llegó en 1942-1943 con batallas decisivas como Stalingrado, 
        donde el ejército alemán sufrió una derrota que marcó el inicio de su retirada."""
        
        try:
            logger.info("🧪 Probando generador con contenido específico de la Segunda Guerra Mundial...")
            
            # Generar quiz usando el servicio
            quiz_result = self.ai_service.test_quiz_generation(wwii_test_text)
            
            return {
                "test_name": "WWII Content Test",
                "test_successful": quiz_result.get("test_successful", False),
                "questions_generated": quiz_result.get("questions_generated", 0),
                "sample_questions": quiz_result.get("sample_questions", []),
                "quality_metrics": quiz_result.get("quality_metrics", {}),
                "first_question_example": quiz_result.get("first_question_detail", {}),
                "text_length": len(wwii_test_text),
                "content_domain": "Segunda Guerra Mundial"
            }
            
        except Exception as e:
            logger.error(f"Error en prueba de WWII: {e}")
            return {
                "test_name": "WWII Content Test",
                "test_successful": False,
                "error": str(e)
            }
    
    def debug_quiz_generation(self, text: str, concepts: List[str] = None) -> dict:
        """
        Función de debugging para analizar el proceso de generación
        """
        if concepts is None:
            concepts = []
        
        debug_info = {
            "input_analysis": {},
            "generation_steps": [],
            "final_result": {}
        }
        
        try:
            # Analizar el input
            debug_info["input_analysis"] = {
                "text_length": len(text),
                "text_preview": text[:200] + "...",
                "concepts_provided": concepts,
                "contains_dates": bool(re.findall(r'\b\d{4}\b', text)),
                "contains_names": bool(re.findall(r'\b[A-Z][a-z]+\b', text))
            }
            
            # Intentar generar quiz y capturar cada paso
            debug_info["generation_steps"].append("Iniciando generación contextual...")
            
            generator = self.ai_service.contextual_quiz_generator
            content_info = generator._extract_content_information(text)
            
            debug_info["generation_steps"].append(f"Dominio detectado: {content_info.get('domain', 'unknown')}")
            debug_info["generation_steps"].append(f"Fechas encontradas: {content_info.get('fechas', [])}")
            debug_info["generation_steps"].append(f"Eventos encontrados: {content_info.get('eventos', [])}")
            debug_info["generation_steps"].append(f"Personajes encontrados: {content_info.get('personajes', [])}")
            
            # Generar quiz
            questions = generator.generate_contextual_quiz(text, 3)  # Solo 3 para debugging
            
            debug_info["final_result"] = {
                "questions_generated": len(questions),
                "questions_preview": [
                    {
                        "question": q.get("question", ""),
                        "category": q.get("category", "unknown"),
                        "source": q.get("source", "unknown")
                    } for q in questions
                ]
            }
            
            return debug_info
            
        except Exception as e:
            debug_info["error"] = str(e)
            debug_info["generation_steps"].append(f"Error: {e}")
            return debug_info

# Instancia global
service_manager = ServiceManager()