# app/services/simple_ai_service.py - SERVICIO SIMPLIFICADO CON QUIZ CONTEXTUAL
import logging
from typing import Dict, Any, List
from app.services.simple_contextual_quiz import SimpleContextualQuizGenerator
from app.services.improved_ai_service import ImprovedAIService

logger = logging.getLogger(__name__)

class SimpleAIServiceWithContextualQuiz(ImprovedAIService):
    """
    Servicio de IA simplificado que usa el generador contextual específico
    """
    
    def __init__(self):
        super().__init__()
        self.contextual_quiz_generator = SimpleContextualQuizGenerator()
        logger.info("✅ SimpleAIServiceWithContextualQuiz inicializado")
    
    async def generate_quiz(self, text: str, key_concepts: List[str], 
                          num_questions: int = 5, difficulty: str = "medium") -> Dict[str, Any]:
        """
        Genera quiz usando el generador contextual específico
        """
        try:
            logger.info(f"🎯 Generando quiz contextual específico con {num_questions} preguntas")
            
            # MÉTODO PRINCIPAL: Usar generador contextual específico
            try:
                contextual_questions = self.contextual_quiz_generator.generate_contextual_quiz(
                    text, num_questions
                )
                
                if contextual_questions and len(contextual_questions) > 0:
                    # Ajustar formato para compatibilidad
                    formatted_questions = []
                    
                    for question in contextual_questions:
                        formatted_question = {
                            "id": question["id"],
                            "question": question["question"],
                            "options": question["options"],
                            "correct_answer": question["correct_answer"],
                            "explanation": question["explanation"],
                            "difficulty": difficulty
                        }
                        formatted_questions.append(formatted_question)
                    
                    logger.info(f"✅ Quiz contextual generado exitosamente con {len(formatted_questions)} preguntas")
                    
                    return {
                        "questions": formatted_questions,
                        "success": True,
                        "generation_method": "contextual_specific",
                        "content_analysis": "text_specific_extraction"
                    }
            
            except Exception as e:
                logger.warning(f"⚠️ Error con generador contextual: {e}")
            
            # MÉTODO DE RESPALDO: Usar método original mejorado
            logger.info("🔄 Usando método de respaldo...")
            return await super().generate_quiz(text, key_concepts, num_questions, difficulty)
            
        except Exception as e:
            logger.error(f"❌ Error crítico generando quiz: {e}")
            return self._generate_emergency_quiz(text, key_concepts, num_questions, difficulty)
    
    def _generate_emergency_quiz(self, text: str, key_concepts: List[str], 
                               num_questions: int, difficulty: str) -> Dict[str, Any]:
        """
        Genera quiz de emergencia cuando fallan otros métodos
        """
        logger.warning("🚨 Generando quiz de emergencia")
        
        # Extraer oraciones específicas del texto
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
        
        questions = []
        
        for i in range(min(num_questions, len(sentences), 5)):
            sentence = sentences[i] if i < len(sentences) else sentences[0]
            
            # Extraer una palabra clave de la oración
            words = [w for w in sentence.split() if len(w) > 4 and w[0].isupper()]
            keyword = words[0] if words else "el tema"
            
            question = {
                "id": i + 1,
                "question": f"Según el texto, ¿qué se menciona sobre {keyword.lower()}?",
                "options": [
                    sentence[:80] + "..." if len(sentence) > 80 else sentence,
                    f"Una información que no aparece en el texto sobre {keyword.lower()}",
                    f"Una interpretación incorrecta sobre {keyword.lower()}",
                    f"Una conclusión no fundamentada sobre {keyword.lower()}"
                ],
                "correct_answer": 0,
                "explanation": f"Esta información aparece directamente en el texto: '{sentence[:100]}...'",
                "difficulty": difficulty
            }
            
            questions.append(question)
        
        return {
            "questions": questions,
            "success": True,
            "generation_method": "emergency_text_based",
            "note": "Quiz generado con extracción directa de oraciones del texto"
        }
    
    def test_quiz_generation(self, test_text: str = None) -> Dict[str, Any]:
        """
        Prueba la generación de quiz con texto específico
        """
        
        if not test_text:
            test_text = """La Segunda Guerra Mundial comenzó el 1 de septiembre de 1939 cuando Alemania invadió Polonia. 
            Adolf Hitler había ordenado esta invasión como parte de su estrategia expansionista. 
            Francia e Inglaterra declararon la guerra a Alemania el 3 de septiembre de 1939.
            
            La estrategia alemana se basaba en el Blitzkrieg o guerra relámpago, que combinaba tanques, 
            aviación y tropas móviles para lograr victorias rápidas.
            
            En 1941, la guerra se globalizó cuando Alemania atacó la Unión Soviética en la Operación Barbarroja
            y Japón atacó Pearl Harbor, llevando a Estados Unidos al conflicto.
            
            El punto de inflexión de la guerra llegó en 1942-1943 con la batalla de Stalingrado, 
            donde el ejército alemán sufrió una derrota decisiva."""
        
        try:
            # Generar quiz de prueba
            result = self.contextual_quiz_generator.generate_contextual_quiz(test_text, 5)
            
            # Analizar calidad
            quality_metrics = self._analyze_test_quality(result, test_text)
            
            return {
                "test_successful": len(result) > 0,
                "questions_generated": len(result),
                "sample_questions": [q["question"] for q in result[:2]],
                "quality_metrics": quality_metrics,
                "first_question_detail": result[0] if result else None
            }
            
        except Exception as e:
            return {
                "test_successful": False,
                "error": str(e)
            }
    
    def _analyze_test_quality(self, questions: List[Dict], text: str) -> Dict[str, Any]:
        """
        Analiza la calidad de las preguntas generadas en la prueba
        """
        if not questions:
            return {"overall_quality": "Sin preguntas generadas"}
        
        metrics = {
            "content_specificity": 0,
            "question_variety": 0,
            "answer_quality": 0
        }
        
        # Evaluar especificidad del contenido
        specific_elements = ["1939", "1945", "Hitler", "Polonia", "Alemania", "Blitzkrieg", "Barbarroja", "Stalingrado"]
        
        for question in questions:
            q_text = question.get("question", "").lower()
            options = question.get("options", [])
            
            # Especificidad del contenido
            if any(element.lower() in q_text for element in specific_elements):
                metrics["content_specificity"] += 1
            
            # Calidad de las respuestas
            correct_option = options[question.get("correct_answer", 0)]
            if any(element.lower() in correct_option.lower() for element in specific_elements):
                metrics["answer_quality"] += 1
        
        # Evaluar variedad de preguntas
        categories = set(q.get("category", "general") for q in questions)
        metrics["question_variety"] = len(categories)
        
        # Calcular puntuaciones
        total_questions = len(questions)
        final_metrics = {
            "content_specificity": metrics["content_specificity"] / total_questions,
            "question_variety": min(metrics["question_variety"] / 3, 1.0),  # Máximo 3 categorías
            "answer_quality": metrics["answer_quality"] / total_questions,
            "overall_score": (metrics["content_specificity"] + metrics["answer_quality"]) / (2 * total_questions)
        }
        
        return final_metrics