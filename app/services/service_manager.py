# app/services/service_manager.py - ACTUALIZADO CON QUIZ INTELIGENTE
import logging
from typing import Optional
from app.services.enhanced_ai_service_integration import EnhancedAIServiceWithQuiz
from app.services.nlp_service import UniversalNLPService
from app.services.quiz_generator import QuizManager

logger = logging.getLogger(__name__)

class ServiceManager:
    """
    Gestor singleton con generador inteligente de quiz contextual
    """
    _instance: Optional['ServiceManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'ServiceManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("🚀 Inicializando ServiceManager con Quiz Inteligente Contextual")
            self._ai_service: Optional[EnhancedAIServiceWithQuiz] = None
            self._nlp_service: Optional[UniversalNLPService] = None
            self._quiz_manager: Optional[QuizManager] = None
            ServiceManager._initialized = True
        else:
            logger.debug("ServiceManager ya inicializado")
    
    @property
    def ai_service(self) -> EnhancedAIServiceWithQuiz:
        """Obtiene instancia singleton de EnhancedAIServiceWithQuiz"""
        if self._ai_service is None:
            logger.info("🤖 Cargando EnhancedAIServiceWithQuiz con generador inteligente...")
            self._ai_service = EnhancedAIServiceWithQuiz()
            logger.info("✅ Servicio de IA mejorado con quiz inteligente cargado")
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
        """Pre-cargar todos los servicios mejorados con quiz inteligente"""
        logger.info("⚡ Pre-cargando servicios con generador inteligente de quiz...")
        _ = self.ai_service
        _ = self.nlp_service  
        _ = self.quiz_manager
        logger.info("🎯 Servicios con quiz inteligente pre-cargados exitosamente")
    
    def get_status(self) -> dict:
        """Obtiene el estado completo incluyendo el generador inteligente"""
        base_status = {
            "ai_service_loaded": self._ai_service is not None,
            "nlp_service_loaded": self._nlp_service is not None,
            "quiz_manager_loaded": self._quiz_manager is not None,
            "service_manager_version": "intelligent_quiz_v3.0",
            "features": [
                "Intelligent contextual quiz generation",
                "Content-specific question extraction", 
                "Enhanced fact-based questions",
                "Multi-method quiz generation",
                "Contextual answer validation",
                "Specialized prompts",
                "Advanced post-processing"
            ]
        }
        
        # Agregar estado detallado del generador inteligente
        if self._ai_service:
            quiz_generator_status = {
                "type": "EnhancedAIServiceWithQuiz",
                "intelligent_quiz_generator": True,
                "contextual_analysis": True,
                "content_extraction": True,
                "multi_method_fallback": True,
                "spacy_nlp": hasattr(self._ai_service.quiz_generator, 'nlp') and self._ai_service.quiz_generator.nlp is not None,
                "domain_patterns": len(getattr(self._ai_service.quiz_generator, 'domain_patterns', {})),
                "question_categories": ["cronologia", "personajes", "geografia", "causas_consecuencias", "conceptos", "contenido_directo"]
            }
            
            base_status["ai_service_details"] = {
                "type": "EnhancedAIServiceWithQuiz",
                "device": getattr(self._ai_service, 'device', 'unknown'),
                "models_loaded": True,
                "quiz_generator": quiz_generator_status
            }
        
        if self._nlp_service:
            base_status["nlp_service_details"] = {
                "type": "UniversalNLPService", 
                "spacy_model": "loaded" if getattr(self._nlp_service, 'nlp', None) else "fallback",
                "domain_patterns": len(getattr(self._nlp_service, 'domain_patterns', {})),
                "universal_processing": True
            }
        
        return base_status
    
    def test_intelligent_quiz_generation(self, test_text: str = None) -> dict:
        """Prueba la generación inteligente de quiz con texto de ejemplo"""
        
        if not test_text:
            test_text = """La Segunda Guerra Mundial (1939-1945) fue el conflicto más devastador de la historia humana. 
            Comenzó con la invasión alemana de Polonia el 1 de septiembre de 1939, cuando Adolf Hitler ordenó el ataque. 
            Este evento llevó a Francia e Inglaterra a declarar la guerra a Alemania el 3 de septiembre de 1939.
            
            La guerra se caracterizó por el uso de la estrategia Blitzkrieg (guerra relámpago) por parte de Alemania, 
            que combinaba tanques, aviación y tropas móviles para lograr victorias rápidas. 
            
            En 1941, la guerra se globalizó con la Operación Barbarroja (invasión de la URSS) y el ataque japonés 
            a Pearl Harbor que llevó a Estados Unidos al conflicto.
            
            El punto de inflexión llegó en 1942-1943 con batallas como Stalingrado y El Alamein, 
            donde los Aliados comenzaron a tomar la iniciativa.
            
            La guerra terminó con la rendición de Alemania el 8 de mayo de 1945 y de Japón el 2 de septiembre de 1945, 
            tras las bombas atómicas en Hiroshima y Nagasaki."""
        
        try:
            logger.info("🧪 Probando generación inteligente de quiz...")
            
            # Extraer conceptos clave
            concepts = self.nlp_service.extract_key_concepts(test_text, max_concepts=8)
            concept_names = [c["concept"] for c in concepts]
            
            # Generar quiz inteligente
            quiz_result = self.ai_service.generate_quiz(
                text=test_text,
                key_concepts=concept_names,
                num_questions=5,
                difficulty="medium"
            )
            
            # Analizar calidad del resultado
            quality_analysis = self._analyze_quiz_quality(quiz_result, test_text)
            
            return {
                "test_successful": quiz_result.get("success", False),
                "generation_method": quiz_result.get("generation_method", "unknown"),
                "questions_generated": len(quiz_result.get("questions", [])),
                "concepts_used": concept_names,
                "quality_score": quality_analysis["overall_score"],
                "quality_details": quality_analysis,
                "sample_questions": [q["question"] for q in quiz_result.get("questions", [])[:2]]
            }
            
        except Exception as e:
            logger.error(f"Error en prueba de quiz inteligente: {e}")
            return {
                "test_successful": False,
                "error": str(e),
                "generation_method": "error"
            }
    
    def _analyze_quiz_quality(self, quiz_result: dict, original_text: str) -> dict:
        """Analiza la calidad del quiz generado"""
        
        questions = quiz_result.get("questions", [])
        quality_metrics = {
            "content_specificity": 0,
            "option_quality": 0,
            "explanation_quality": 0,
            "question_variety": 0,
            "contextual_relevance": 0
        }
        
        if not questions:
            return {"overall_score": 0, "metrics": quality_metrics}
        
        # Analizar especificidad del contenido
        specific_content_score = 0
        for question in questions:
            q_text = question.get("question", "").lower()
            
            # Buscar elementos específicos del texto
            if any(word in q_text for word in ["1939", "1945", "hitler", "alemania", "polonia"]):
                specific_content_score += 1
            elif any(word in q_text for word in ["segunda guerra", "blitzkrieg", "barbarroja"]):
                specific_content_score += 0.8
            elif "texto" not in q_text and "información" not in q_text:
                specific_content_score += 0.5
        
        quality_metrics["content_specificity"] = specific_content_score / len(questions)
        
        # Analizar calidad de opciones
        option_quality_score = 0
        for question in questions:
            options = question.get("options", [])
            if len(options) == 4:
                # Verificar que las opciones no sean todas genéricas
                generic_options = sum(1 for opt in options if "información" in opt.lower() or "texto" in opt.lower())
                option_quality_score += max(0, (4 - generic_options) / 4)
        
        quality_metrics["option_quality"] = option_quality_score / len(questions) if questions else 0
        
        # Analizar calidad de explicaciones
        explanation_quality_score = 0
        for question in questions:
            explanation = question.get("explanation", "")
            if len(explanation) > 50 and "texto" in explanation and "específica" in explanation:
                explanation_quality_score += 1
            elif len(explanation) > 30:
                explanation_quality_score += 0.6
        
        quality_metrics["explanation_quality"] = explanation_quality_score / len(questions) if questions else 0
        
        # Analizar variedad de preguntas
        categories = set(q.get("category", "unknown") for q in questions)
        quality_metrics["question_variety"] = min(len(categories) / 3, 1.0)  # Máximo 3 categorías esperadas
        
        # Analizar relevancia contextual
        contextual_score = 0
        for question in questions:
            if question.get("source_context") or question.get("source_fact"):
                contextual_score += 1
            elif question.get("enhancement_applied"):
                contextual_score += 0.7
            elif question.get("generation_method") in ["enhanced_contextual", "intelligent_generation"]:
                contextual_score += 0.8
        
        quality_metrics["contextual_relevance"] = contextual_score / len(questions) if questions else 0
        
        # Calcular puntuación general
        overall_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        return {
            "overall_score": round(overall_score, 2),
            "metrics": {k: round(v, 2) for k, v in quality_metrics.items()},
            "quality_level": "Excelente" if overall_score >= 0.8 else "Bueno" if overall_score >= 0.6 else "Básico" if overall_score >= 0.4 else "Necesita mejora"
        }

# Instancia global mejorada
service_manager = ServiceManager()