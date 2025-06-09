# app/services/enhanced_ai_service_integration.py - INTEGRACIÓN COMPLETA
import logging
from typing import Dict, Any, List
from app.services.improved_ai_service import ImprovedAIService
from app.services.improved_quiz_generator import IntelligentQuizGenerator

logger = logging.getLogger(__name__)

class EnhancedAIServiceWithQuiz(ImprovedAIService):
    """
    Servicio de IA mejorado que integra el generador inteligente de quiz
    """
    
    def __init__(self):
        super().__init__()
        self.quiz_generator = IntelligentQuizGenerator()
        logger.info("✅ EnhancedAIServiceWithQuiz inicializado con generador inteligente")
    
    async def generate_quiz(self, text: str, key_concepts: List[str], 
                          num_questions: int = 5, difficulty: str = "medium") -> Dict[str, Any]:
        """
        Genera quiz mejorado usando el generador inteligente contextual
        """
        try:
            logger.info(f"🎯 Generando quiz inteligente con {num_questions} preguntas sobre: {key_concepts[:3]}")
            
            # MÉTODO 1: Usar el generador inteligente contextual (PRINCIPAL)
            try:
                intelligent_result = self.quiz_generator.generate_quiz_with_enhanced_context(
                    text, key_concepts, num_questions, difficulty
                )
                
                if intelligent_result["success"] and len(intelligent_result["questions"]) >= num_questions:
                    logger.info("✅ Quiz generado exitosamente con método inteligente contextual")
                    return intelligent_result
                
            except Exception as e:
                logger.warning(f"⚠️ Error con generador inteligente: {e}")
            
            # MÉTODO 2: Usar método mejorado original como respaldo
            try:
                logger.info("🔄 Usando método mejorado como respaldo...")
                improved_result = await super().generate_quiz(text, key_concepts, num_questions, difficulty)
                
                if improved_result["success"]:
                    # Mejorar las preguntas del método original con contexto
                    enhanced_questions = self._enhance_with_contextual_analysis(
                        improved_result["questions"], text, key_concepts
                    )
                    
                    improved_result["questions"] = enhanced_questions
                    improved_result["generation_method"] = "improved_with_enhancement"
                    
                    logger.info("✅ Quiz generado con método mejorado + mejoras contextuales")
                    return improved_result
                    
            except Exception as e:
                logger.warning(f"⚠️ Error con método mejorado: {e}")
            
            # MÉTODO 3: Fallback final
            return self._generate_enhanced_fallback_quiz(text, key_concepts, num_questions, difficulty)
            
        except Exception as e:
            logger.error(f"❌ Error crítico generando quiz: {e}")
            return {
                "questions": [],
                "success": False,
                "error": str(e),
                "generation_method": "error"
            }
    
    def _enhance_with_contextual_analysis(self, questions: List[Dict], text: str, 
                                        key_concepts: List[str]) -> List[Dict]:
        """
        Mejora preguntas existentes con análisis contextual
        """
        enhanced_questions = []
        
        for question in questions:
            try:
                enhanced_question = self._apply_contextual_improvements(question, text, key_concepts)
                enhanced_questions.append(enhanced_question)
            except Exception as e:
                logger.warning(f"Error mejorando pregunta: {e}")
                enhanced_questions.append(question)  # Mantener original si falla
        
        return enhanced_questions
    
    def _apply_contextual_improvements(self, question: Dict, text: str, 
                                     key_concepts: List[str]) -> Dict:
        """
        Aplica mejoras contextuales a una pregunta específica
        """
        improved_question = question.copy()
        
        # 1. Mejorar la pregunta para que sea más específica del contenido
        improved_question["question"] = self._make_question_more_specific(
            question["question"], text, key_concepts
        )
        
        # 2. Mejorar opciones para que sean más contextuales
        improved_question["options"] = self._improve_options_with_context(
            question["options"], text, question["correct_answer"]
        )
        
        # 3. Generar explicación más detallada y contextual
        improved_question["explanation"] = self._generate_detailed_explanation(
            improved_question, text, key_concepts
        )
        
        # 4. Agregar metadatos de mejora
        improved_question["enhancement_applied"] = True
        improved_question["content_source"] = "contextual_analysis"
        
        return improved_question
    
    def _make_question_more_specific(self, generic_question: str, text: str, 
                                   key_concepts: List[str]) -> str:
        """
        Hace la pregunta más específica del contenido analizado
        """
        
        # Si la pregunta es muy genérica, hacerla más específica
        if "tema principal" in generic_question.lower():
            # Intentar identificar el tema específico del texto
            if "segunda guerra mundial" in text.lower():
                return generic_question.replace("tema principal", "Segunda Guerra Mundial")
            elif "primera guerra mundial" in text.lower():
                return generic_question.replace("tema principal", "Primera Guerra Mundial")
            elif key_concepts:
                return generic_question.replace("tema principal", key_concepts[0])
        
        # Si menciona conceptos genéricos, usar conceptos específicos
        if "concepto" in generic_question.lower() and key_concepts:
            for concept in key_concepts:
                if concept.lower() in text.lower():
                    return generic_question.replace("concepto", concept)
        
        return generic_question
    
    def _improve_options_with_context(self, options: List[str], text: str, 
                                    correct_answer_index: int) -> List[str]:
        """
        Mejora las opciones usando información específica del texto
        """
        improved_options = []
        
        for i, option in enumerate(options):
            if i == correct_answer_index:
                # Para la opción correcta, intentar usar información más específica del texto
                improved_option = self._make_correct_option_more_specific(option, text)
                improved_options.append(improved_option)
            else:
                # Para opciones incorrectas, mantenerlas pero hacerlas más específicas si es posible
                improved_option = self._improve_distractor_option(option, text)
                improved_options.append(improved_option)
        
        return improved_options
    
    def _make_correct_option_more_specific(self, correct_option: str, text: str) -> str:
        """
        Hace la opción correcta más específica usando información del texto
        """
        
        # Si la opción es muy genérica, intentar usar información específica del texto
        if "información" in correct_option.lower() and "texto" in correct_option.lower():
            
            # Extraer oraciones informativas del texto
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 30]
            
            if sentences:
                # Tomar la primera oración informativa
                specific_info = sentences[0]
                if len(specific_info) > 100:
                    specific_info = specific_info[:100] + "..."
                
                return specific_info
        
        return correct_option
    
    def _improve_distractor_option(self, distractor: str, text: str) -> str:
        """
        Mejora un distractor para que sea más específico pero siga siendo incorrecto
        """
        
        # Lista de distractores mejorados y específicos
        specific_distractors = [
            "Una interpretación que no está respaldada por las fuentes históricas mencionadas",
            "Una información que contradice los datos específicos presentados en el texto",
            "Una conclusión que no se deriva del análisis del contenido estudiado",
            "Una afirmación que no tiene fundamento en la documentación analizada",
            "Una perspectiva que no coincide con los hechos históricos presentados"
        ]
        
        # Si el distractor es muy genérico, usar uno más específico
        if any(word in distractor.lower() for word in ["información", "conclusión", "interpretación"]):
            return random.choice(specific_distractors)
        
        return distractor
    
    def _generate_detailed_explanation(self, question: Dict, text: str, 
                                     key_concepts: List[str]) -> str:
        """
        Genera explicación detallada y contextual
        """
        
        correct_option = question["options"][question["correct_answer"]]
        question_text = question["question"]
        
        # Buscar evidencia específica en el texto
        evidence = self._find_textual_evidence(correct_option, text)
        
        if evidence:
            return f"La respuesta correcta es '{correct_option[:60]}...' porque el texto específicamente menciona: '{evidence[:100]}...'. Esta información es clave para entender el tema analizado."
        
        # Explicación de respaldo
        main_concept = key_concepts[0] if key_concepts else "el tema principal"
        
        return f"La respuesta correcta es '{correct_option[:60]}...' según la información presentada en el texto sobre {main_concept}. Esta respuesta se basa en el contenido específico analizado y su contexto histórico."
    
    def _find_textual_evidence(self, correct_option: str, text: str) -> str:
        """
        Busca evidencia textual específica para una respuesta
        """
        
        # Extraer palabras clave de la opción correcta
        option_words = [word.lower() for word in correct_option.split() if len(word) > 3]
        
        # Buscar en oraciones del texto
        sentences = re.split(r'[.!?]+', text)
        
        best_match = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Contar coincidencias de palabras clave
            matches = sum(1 for word in option_words if word in sentence_lower)
            
            if matches > best_score and len(sentence.strip()) > 30:
                best_score = matches
                best_match = sentence.strip()
        
        return best_match if best_score >= 2 else ""
    
    def _generate_enhanced_fallback_quiz(self, text: str, key_concepts: List[str], 
                                       num_questions: int, difficulty: str) -> Dict[str, Any]:
        """
        Genera quiz de respaldo mejorado cuando fallan otros métodos
        """
        
        logger.info("🔄 Generando quiz de respaldo mejorado...")
        
        questions = []
        
        # Usar análisis básico del texto para crear preguntas más específicas
        text_analysis = self._analyze_text_for_fallback_questions(text)
        
        for i in range(num_questions):
            question = self._create_enhanced_fallback_question(
                i + 1, text, key_concepts, text_analysis, difficulty
            )
            questions.append(question)
        
        return {
            "questions": questions,
            "success": True,
            "generation_method": "enhanced_fallback",
            "note": "Quiz generado con análisis básico mejorado del contenido"
        }
    
    def _analyze_text_for_fallback_questions(self, text: str) -> Dict[str, Any]:
        """
        Analiza el texto para extraer información básica para preguntas de respaldo
        """
        
        analysis = {
            "key_sentences": [],
            "important_dates": [],
            "key_figures": [],
            "main_topics": [],
            "factual_statements": []
        }
        
        # Extraer oraciones importantes
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 40]
        analysis["key_sentences"] = sentences[:5]
        
        # Extraer fechas
        dates = re.findall(r'\b\d{4}\b', text)
        analysis["important_dates"] = list(set(dates))[:3]
        
        # Extraer figuras importantes (nombres propios)
        figures = re.findall(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)*\b', text)
        analysis["key_figures"] = list(set(figures))[:5]
        
        # Identificar temas principales
        if "segunda guerra mundial" in text.lower():
            analysis["main_topics"].append("Segunda Guerra Mundial")
        if "hitler" in text.lower():
            analysis["main_topics"].append("Adolf Hitler")
        if "alemania" in text.lower():
            analysis["main_topics"].append("Alemania")
        
        return analysis
    
    def _create_enhanced_fallback_question(self, question_id: int, text: str, 
                                         key_concepts: List[str], analysis: Dict, 
                                         difficulty: str) -> Dict:
        """
        Crea pregunta de respaldo mejorada usando análisis del texto
        """
        
        # Usar información del análisis para crear preguntas más específicas
        if analysis["important_dates"] and question_id <= 2:
            # Pregunta sobre fechas
            date = analysis["important_dates"][0]
            question_text = f"¿En qué año se menciona que ocurrieron eventos importantes según el texto?"
            
            correct_answer = date
            options = [
                correct_answer,
                str(int(date) - 1),
                str(int(date) + 1),
                str(int(date) - 2)
            ]
            
        elif analysis["key_figures"] and question_id <= 3:
            # Pregunta sobre figuras
            figure = analysis["key_figures"][0]
            question_text = f"¿Qué figura histórica se menciona como relevante en el texto?"
            
            options = [
                figure,
                "Una figura que no se menciona en el texto",
                "Un personaje secundario no identificado",
                "Una personalidad no relacionada con el tema"
            ]
            correct_answer = 0
            
        elif analysis["key_sentences"]:
            # Pregunta sobre contenido específico
            key_sentence = analysis["key_sentences"][question_id % len(analysis["key_sentences"])]
            question_text = "¿Cuál de las siguientes afirmaciones corresponde al contenido del texto?"
            
            # Usar parte de la oración como respuesta correcta
            correct_answer_text = key_sentence[:80] + "..." if len(key_sentence) > 80 else key_sentence
            
            options = [
                correct_answer_text,
                "Una afirmación que contradice la información del texto",
                "Una interpretación que no tiene base en el contenido",
                "Una conclusión que no se deriva del análisis presentado"
            ]
            correct_answer = 0
            
        else:
            # Pregunta genérica pero mejorada
            main_concept = key_concepts[0] if key_concepts else "el tema principal"
            question_text = f"Según el análisis del texto, ¿cuál es la información más precisa sobre {main_concept}?"
            
            options = [
                f"La información específica que se presenta en el texto sobre {main_concept}",
                f"Una interpretación general no basada en el contenido específico",
                f"Una conclusión que no está respaldada por el texto",
                f"Una afirmación que contradice lo expuesto en el contenido"
            ]
            correct_answer = 0
        
        # Mezclar opciones si no son fechas
        if not analysis["important_dates"] or question_id > 2:
            random.shuffle(options)
            correct_answer = options.index([opt for opt in options if "específica" in opt or "presenta" in opt][0]) if any("específica" in opt or "presenta" in opt for opt in options) else 0
        
        explanation = f"La respuesta correcta se basa en la información específica que aparece en el texto analizado. El contenido proporciona datos precisos sobre este aspecto del tema."
        
        return {
            "id": question_id,
            "question": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": explanation,
            "difficulty": difficulty,
            "category": "enhanced_fallback",
            "source": "text_analysis"
        }

# Importar re para el procesamiento de texto
import re
import random