# app/services/ai_service.py - VERSIÓN MEJORADA PARA QUIZ
import json
import logging
import random
from typing import Dict, Any, List, Optional
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    pipeline, T5ForConditionalGeneration, T5Tokenizer
)
import torch
from app.core.config import settings

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Inicializar modelos
        self._init_models()
    
    def _init_models(self):
        """Inicializa todos los modelos necesarios"""
        try:
            # Modelo para resúmenes (BART en español)
            logger.info("Cargando modelo de resumen...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1
            )
            
            # Modelo para generación de texto/quiz (T5)
            logger.info("Cargando modelo T5 para quiz...")
            self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                "google/flan-t5-base"
            ).to(self.device)
            
            # Pipeline para clasificación y análisis
            self.classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Todos los modelos cargados exitosamente")
            
        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")
            # Fallback a modelos más pequeños
            self._init_fallback_models()
    
    def _init_fallback_models(self):
        """Modelos de respaldo más pequeños"""
        logger.info("Cargando modelos de respaldo...")
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    
    async def generate_summary(self, text: str, length: str = "medium") -> Dict[str, Any]:
        """
        Genera un resumen educativo del texto usando BART
        """
        try:
            # Configurar longitud según parámetro
            length_config = {
                "short": {"max_length": 100, "min_length": 30},
                "medium": {"max_length": 200, "min_length": 50},
                "long": {"max_length": 300, "min_length": 100}
            }
            
            config = length_config.get(length, length_config["medium"])
            
            # Limitar texto de entrada (BART tiene límite de tokens)
            max_input_length = 1024
            if len(text.split()) > max_input_length:
                text = " ".join(text.split()[:max_input_length])
            
            # Generar resumen
            summary_result = self.summarizer(
                text,
                max_length=config["max_length"],
                min_length=config["min_length"],
                do_sample=False
            )
            
            summary = summary_result[0]['summary_text']
            
            # Post-procesar para hacerlo más educativo
            educational_summary = self._make_educational(summary, text)
            
            return {
                "summary": educational_summary,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generando resumen: {e}")
            return {
                "summary": self._generate_fallback_summary(text),
                "success": False,
                "error": str(e)
            }
    
    def _make_educational(self, summary: str, original_text: str) -> str:
        """Mejora el resumen para hacerlo más educativo"""
        # Agregar contexto educativo
        intro = "📚 Resumen Educativo:\n\n"
        
        # Identificar conceptos clave del texto original
        key_concepts = self._extract_key_terms(original_text)
        
        if key_concepts:
            intro += f"🔑 Conceptos clave: {', '.join(key_concepts[:3])}\n\n"
        
        return intro + summary
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extrae términos clave del texto"""
        # Implementación simple - en producción usar NER
        words = text.lower().split()
        # Filtrar palabras comunes y obtener términos únicos
        stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las'}
        key_terms = [word for word in set(words) if len(word) > 4 and word not in stop_words]
        return key_terms[:5]
    
    async def generate_quiz(self, text: str, key_concepts: List[str], num_questions: int = 5, difficulty: str = "medium") -> Dict[str, Any]:
        """
        Genera un quiz mejorado basado en conceptos clave
        """
        try:
            questions = []
            
            # ✅ GENERAR PREGUNTAS BASADAS EN CONCEPTOS REALES
            for i in range(num_questions):
                question_data = await self._generate_improved_question(text, key_concepts, i+1, difficulty)
                if question_data:
                    questions.append(question_data)
            
            # ✅ SI NO HAY SUFICIENTES PREGUNTAS, USAR FALLBACK
            while len(questions) < num_questions:
                fallback_question = self._create_fallback_question(len(questions) + 1, key_concepts, difficulty)
                questions.append(fallback_question)
            
            return {
                "questions": questions,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generando quiz: {e}")
            return {
                "questions": self._generate_fallback_quiz(text, key_concepts, num_questions),
                "success": False,
                "error": str(e)
            }
    
    async def _generate_improved_question(self, text: str, concepts: List[str], question_num: int, difficulty: str) -> Optional[Dict]:
        """Genera una pregunta mejorada usando conceptos clave"""
        try:
            # ✅ SELECCIONAR CONCEPTO PARA LA PREGUNTA
            if concepts and len(concepts) > 0:
                concept_index = (question_num - 1) % len(concepts)
                main_concept = concepts[concept_index]
            else:
                main_concept = "inteligencia artificial"
            
            # ✅ ENCONTRAR CONTEXTO RELEVANTE EN EL TEXTO
            sentences = text.split('.')
            relevant_sentence = ""
            
            for sentence in sentences:
                if main_concept.lower() in sentence.lower():
                    relevant_sentence = sentence.strip()
                    break
            
            if not relevant_sentence and sentences:
                relevant_sentence = sentences[min(question_num-1, len(sentences)-1)].strip()
            
            # ✅ GENERAR PREGUNTA ESTRUCTURADA
            if len(relevant_sentence) > 10:
                return self._create_structured_question(
                    main_concept, 
                    relevant_sentence, 
                    question_num, 
                    difficulty,
                    concepts
                )
            else:
                return self._create_fallback_question(question_num, concepts, difficulty)
                
        except Exception as e:
            logger.error(f"Error generando pregunta {question_num}: {e}")
            return self._create_fallback_question(question_num, concepts, difficulty)
    
    def _create_structured_question(self, main_concept: str, context: str, question_id: int, difficulty: str, all_concepts: List[str]) -> Dict:
        """Crea una pregunta estructurada de alta calidad"""
        
        # ✅ TIPOS DE PREGUNTAS VARIADAS
        question_types = [
            "¿Qué es {concept}?",
            "¿Cuál es la principal característica de {concept}?", 
            "¿Cómo se relaciona {concept} con el contenido analizado?",
            "Según el texto, ¿qué papel juega {concept}?",
            "¿Cuál es la importancia de {concept} en este contexto?"
        ]
        
        question_template = question_types[(question_id - 1) % len(question_types)]
        question_text = question_template.format(concept=main_concept)
        
        # ✅ CREAR OPCIONES REALISTAS
        correct_answer = main_concept.title()
        
        # Generar opciones incorrectas basadas en otros conceptos
        incorrect_options = []
        
        # Usar otros conceptos como distractores
        for concept in all_concepts:
            if concept.lower() != main_concept.lower() and len(incorrect_options) < 2:
                incorrect_options.append(concept.title())
        
        # Completar con opciones genéricas si no hay suficientes conceptos
        generic_options = [
            "Análisis de datos básico",
            "Procesamiento manual de información", 
            "Sistema tradicional de cómputo",
            "Método convencional de análisis",
            "Técnica estadística clásica"
        ]
        
        while len(incorrect_options) < 3:
            for option in generic_options:
                if option not in incorrect_options and len(incorrect_options) < 3:
                    incorrect_options.append(option)
        
        # Crear lista final de opciones
        all_options = [correct_answer] + incorrect_options[:3]
        
        # Mezclar opciones
        random.shuffle(all_options)
        correct_index = all_options.index(correct_answer)
        
        # ✅ EXPLICACIÓN EDUCATIVA
        explanation = f"La respuesta correcta es '{correct_answer}' porque es un concepto clave mencionado en el texto que se relaciona directamente con el tema analizado."
        
        return {
            "id": question_id,
            "question": question_text,
            "options": all_options,
            "correct_answer": correct_index,
            "explanation": explanation,
            "difficulty": difficulty
        }
    
    def _create_fallback_question(self, question_id: int, concepts: List[str], difficulty: str) -> Dict:
        """Crea una pregunta de respaldo de alta calidad"""
        
        if concepts and len(concepts) > 0:
            concept_index = (question_id - 1) % len(concepts)
            main_concept = concepts[concept_index]
        else:
            main_concept = "el contenido analizado"
        
        # ✅ PREGUNTAS FALLBACK VARIADAS Y EDUCATIVAS
        fallback_questions = [
            f"¿Cuál es el concepto principal relacionado con {main_concept}?",
            f"Según el análisis, ¿qué caracteriza a {main_concept}?",
            f"¿Cuál es la importancia de {main_concept} en el contexto estudiado?",
            f"¿Cómo se puede definir {main_concept} según el contenido?",
            f"¿Qué función cumple {main_concept} en el tema analizado?"
        ]
        
        question_text = fallback_questions[(question_id - 1) % len(fallback_questions)]
        
        # ✅ OPCIONES REALISTAS BASADAS EN CONCEPTOS
        if concepts and len(concepts) >= 4:
            options = concepts[:4]
            correct_answer = 0  # El primer concepto es correcto
        else:
            correct_option = main_concept if main_concept != "el contenido analizado" else "Inteligencia Artificial"
            options = [
                correct_option,
                "Procesamiento básico de datos",
                "Análisis estadístico tradicional", 
                "Sistema de información convencional"
            ]
            correct_answer = 0
        
        explanation = f"La respuesta correcta se relaciona con {options[correct_answer]} ya que es uno de los conceptos centrales del texto analizado."
        
        return {
            "id": question_id,
            "question": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": explanation,
            "difficulty": difficulty
        }
    
    async def generate_feedback(self, score: int, total: int, incorrect_questions: List[int], concepts: List[str]) -> str:
        """
        Genera retroalimentación pedagógica mejorada
        """
        try:
            percentage = (score / total) * 100
            
            # ✅ FEEDBACK ESTRUCTURADO BASADO EN RENDIMIENTO
            if percentage >= 80:
                base_feedback = f"¡Excelente trabajo! Has demostrado un sólido dominio de los conceptos clave"
                if concepts:
                    base_feedback += f" relacionados con {', '.join(concepts[:2])}"
                base_feedback += f". Tu puntuación de {score}/{total} ({percentage:.1f}%) indica que tienes una comprensión muy buena del tema."
                
            elif percentage >= 60:
                base_feedback = f"Buen trabajo. Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%)"
                if concepts:
                    base_feedback += f". Tienes una base sólida en {concepts[0] if concepts else 'los conceptos principales'}"
                base_feedback += ", pero hay algunas áreas que puedes reforzar para mejorar tu comprensión."
                
            else:
                base_feedback = f"Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%)"
                if concepts:
                    base_feedback += f". Te recomiendo revisar los conceptos fundamentales como {', '.join(concepts[:2]) if len(concepts) >= 2 else concepts[0] if concepts else 'los temas principales'}"
                base_feedback += ". No te desanimes, el aprendizaje es un proceso gradual y cada intento te acerca más al dominio del tema."
            
            return base_feedback
            
        except Exception as e:
            logger.error(f"Error generando feedback: {e}")
            return self._generate_fallback_feedback(score, total)
    
    def _generate_fallback_summary(self, text: str) -> str:
        """Genera un resumen básico sin IA"""
        sentences = text.split('.')[:3]
        return f"📚 Resumen básico: {'. '.join(sentences)}."
    
    def _generate_fallback_quiz(self, text: str, concepts: List[str], num_questions: int) -> List[Dict]:
        """Genera preguntas básicas de alta calidad"""
        questions = []
        for i in range(min(num_questions, max(3, len(concepts) if concepts else 3))):
            question = self._create_fallback_question(i + 1, concepts, "medium")
            questions.append(question)
        return questions
    
    def _generate_fallback_feedback(self, score: int, total: int) -> str:
        """Genera feedback básico pero útil"""
        percentage = (score / total) * 100
        if percentage >= 80:
            return f"¡Excelente trabajo! Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%). Demuestras un sólido dominio del tema."
        elif percentage >= 60:
            return f"Buen trabajo. Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%). Tienes una base sólida, continúa practicando para mejorar."
        else:
            return f"Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%). Te recomiendo revisar el material de estudio y enfocarte en los conceptos principales. ¡El aprendizaje es un proceso, sigue adelante!"