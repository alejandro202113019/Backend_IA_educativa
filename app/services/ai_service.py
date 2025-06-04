# app/services/ai_service.py - VERSIÓN CON MODELOS GRATUITOS
import json
import logging
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
        Genera un quiz usando T5
        """
        try:
            questions = []
            
            for i in range(num_questions):
                question_data = await self._generate_single_question(text, key_concepts, i+1, difficulty)
                if question_data:
                    questions.append(question_data)
            
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
    
    async def _generate_single_question(self, text: str, concepts: List[str], question_num: int, difficulty: str) -> Optional[Dict]:
        """Genera una pregunta individual usando T5"""
        try:
            # Tomar una sección del texto para la pregunta
            sentences = text.split('.')
            if len(sentences) > question_num:
                context = '. '.join(sentences[question_num-1:question_num+1])
            else:
                context = text[:200]  # Primeros 200 caracteres
            
            # Prompt para T5
            prompt = f"Genera una pregunta de opción múltiple sobre: {context}"
            
            # Tokenizar y generar
            inputs = self.t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    inputs, 
                    max_length=200, 
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            generated_text = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Crear estructura de pregunta
            question_data = self._parse_generated_question(generated_text, question_num, difficulty, context, concepts)
            
            return question_data
            
        except Exception as e:
            logger.error(f"Error generando pregunta {question_num}: {e}")
            return self._create_fallback_question(question_num, concepts, difficulty)
    
    def _parse_generated_question(self, generated_text: str, question_id: int, difficulty: str, context: str, concepts: List[str]) -> Dict:
        """Parsea la respuesta generada y crea una pregunta estructurada"""
        # Si la generación no es buena, crear pregunta basada en conceptos
        if len(generated_text) < 10:
            return self._create_fallback_question(question_id, concepts, difficulty)
        
        # Usar el texto generado como base para la pregunta
        question_text = f"Según el contenido analizado, {generated_text}"
        
        # Crear opciones basadas en conceptos
        if concepts and len(concepts) >= 1:
            correct_concept = concepts[0] if concepts else "concepto clave"
            options = [
                correct_concept,
                "Información incorrecta A",
                "Información incorrecta B", 
                "Información incorrecta C"
            ]
            
            # Mezclar opciones
            import random
            correct_index = random.randint(0, 3)
            options[0], options[correct_index] = options[correct_index], options[0]
            
            return {
                "id": question_id,
                "question": question_text,
                "options": options,
                "correct_answer": correct_index,
                "explanation": f"La respuesta correcta se relaciona con {correct_concept} mencionado en el texto.",
                "difficulty": difficulty
            }
        
        return self._create_fallback_question(question_id, concepts, difficulty)
    
    def _create_fallback_question(self, question_id: int, concepts: List[str], difficulty: str) -> Dict:
        """Crea una pregunta de respaldo"""
        if concepts and len(concepts) > 0:
            concept = concepts[min(question_id-1, len(concepts)-1)]
        else:
            concept = "el contenido analizado"
        
        return {
            "id": question_id,
            "question": f"¿Cuál es el concepto principal relacionado con {concept}?",
            "options": [
                concept if concepts else "Concepto principal",
                "Opción incorrecta 1",
                "Opción incorrecta 2",
                "Opción incorrecta 3"
            ],
            "correct_answer": 0,
            "explanation": f"La respuesta correcta es {concept} ya que es uno de los conceptos centrales del texto analizado.",
            "difficulty": difficulty
        }
    
    async def generate_feedback(self, score: int, total: int, incorrect_questions: List[int], concepts: List[str]) -> str:
        """
        Genera retroalimentación pedagógica usando T5
        """
        try:
            percentage = (score / total) * 100
            
            # Crear prompt para retroalimentación
            prompt = f"Genera retroalimentación educativa para un estudiante que obtuvo {score} de {total} preguntas correctas ({percentage:.1f}%). Conceptos: {', '.join(concepts[:3])}"
            
            inputs = self.t5_tokenizer.encode(prompt, return_tensors="pt", max_length=256, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    inputs,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            feedback = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Agregar contexto adicional
            if percentage >= 80:
                feedback = f"¡Excelente trabajo! {feedback} Continúa así."
            elif percentage >= 60:
                feedback = f"Buen esfuerzo. {feedback} Sigue practicando."
            else:
                feedback = f"Te recomiendo repasar el material. {feedback} ¡No te rindas!"
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error generando feedback: {e}")
            return self._generate_fallback_feedback(score, total)
    
    def _generate_fallback_summary(self, text: str) -> str:
        """Genera un resumen básico sin IA"""
        sentences = text.split('.')[:3]
        return f"📚 Resumen básico: {'. '.join(sentences)}."
    
    def _generate_fallback_quiz(self, text: str, concepts: List[str], num_questions: int) -> List[Dict]:
        """Genera preguntas básicas sin IA"""
        questions = []
        for i in range(min(num_questions, max(3, len(concepts)))):
            concept = concepts[i] if i < len(concepts) else f"Concepto {i+1}"
            questions.append({
                "id": i + 1,
                "question": f"¿Cuál es el concepto clave relacionado con {concept}?",
                "options": [concept, "Opción B", "Opción C", "Opción D"],
                "correct_answer": 0,
                "explanation": f"La respuesta correcta es {concept} según el texto analizado.",
                "difficulty": "medium"
            })
        return questions
    
    def _generate_fallback_feedback(self, score: int, total: int) -> str:
        """Genera feedback básico sin IA"""
        percentage = (score / total) * 100
        if percentage >= 80:
            return f"¡Excelente trabajo! Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%). Dominas bien el tema."
        elif percentage >= 60:
            return f"Buen trabajo. Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%). Continúa estudiando para mejorar."
        else:
            return f"Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%). Te recomiendo revisar el material de estudio nuevamente."