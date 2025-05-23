# app/services/ai_service.py CORREGIDO (OpenAI v1+ compatible)
import openai
import json
import logging
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.models.response_models import QuizQuestion

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        if settings.OPENAI_API_KEY:
            self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)  # ← CAMBIO AQUÍ
        else:
            self.client = None
            logger.warning("OpenAI API key not configured")
    
    async def generate_summary(self, text: str, length: str = "medium") -> Dict[str, Any]:
        """
        Genera un resumen educativo del texto
        """
        if not self.client:
            return {
                "summary": self._generate_fallback_summary(text),
                "success": False,
                "error": "OpenAI API key not configured"
            }
        
        try:
            # Determinar longitud del resumen
            length_prompts = {
                "short": "Genera un resumen muy conciso en 1 párrafo",
                "medium": "Genera un resumen educativo en 2-3 párrafos",
                "long": "Genera un resumen detallado y educativo en 4-5 párrafos"
            }
            
            prompt = f"""
            {length_prompts.get(length, length_prompts["medium"])} del siguiente texto para estudiantes:

            INSTRUCCIONES:
            1. Use un tono didáctico pero cercano
            2. Destaque 3-5 conceptos clave más importantes
            3. Incluya ejemplos prácticos cuando sea posible
            4. Estructure como: [Contexto] [Conceptos Principales] [Aplicación/Relevancia]
            5. Escriba en español
            6. Sea educativo, no solo informativo

            TEXTO A RESUMIR:
            {text[:4000]}

            RESUMEN EDUCATIVO:
            """

            response = self.client.chat.completions.create(  # ← CAMBIO AQUÍ
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente educativo especializado en crear resúmenes didácticos para estudiantes."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            summary = response.choices[0].message.content.strip()
            
            return {
                "summary": summary,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generando resumen: {e}")
            return {
                "summary": self._generate_fallback_summary(text),
                "success": False,
                "error": str(e)
            }
    
    async def generate_quiz(self, text: str, key_concepts: List[str], num_questions: int = 5, difficulty: str = "medium") -> Dict[str, Any]:
        """
        Genera un quiz educativo basado en el texto y conceptos clave
        """
        if not self.client:
            return {
                "questions": self._generate_fallback_quiz(text, key_concepts, num_questions),
                "success": False,
                "error": "OpenAI API key not configured"
            }
        
        try:
            difficulty_settings = {
                "easy": "preguntas básicas de comprensión",
                "medium": "preguntas de análisis y aplicación", 
                "hard": "preguntas de síntesis y evaluación crítica"
            }
            
            concepts_str = ", ".join(key_concepts[:8])
            
            prompt = f"""
            Genera {num_questions} preguntas de múltiple opción ({difficulty_settings.get(difficulty, "medium")}) sobre el siguiente contenido educativo:

            CONCEPTOS CLAVE: {concepts_str}

            TEXTO DE REFERENCIA:
            {text[:3000]}

            FORMATO REQUERIDO (JSON):
            {{
                "questions": [
                    {{
                        "id": 1,
                        "question": "Pregunta aquí",
                        "options": ["Opción A", "Opción B", "Opción C", "Opción D"],
                        "correct_answer": 0,
                        "explanation": "Explicación pedagógica de por qué esta es la respuesta correcta",
                        "difficulty": "{difficulty}"
                    }}
                ]
            }}

            REGLAS:
            1. 4 opciones por pregunta
            2. Solo UNA respuesta correcta por pregunta
            3. Explicaciones pedagógicas claras
            4. Preguntas educativas, no triviales
            5. Dificultad progresiva
            6. En español
            7. Responde SOLO con JSON válido
            """

            response = self.client.chat.completions.create(  # ← CAMBIO AQUÍ
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un experto en evaluación educativa. Genera solo JSON válido."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.8
            )
            
            quiz_content = response.choices[0].message.content.strip()
            
            # Limpiar posible texto extra
            if "```json" in quiz_content:
                quiz_content = quiz_content.split("```json")[1].split("```")[0]
            elif "```" in quiz_content:
                quiz_content = quiz_content.split("```")[1].split("```")[0]
            
            quiz_data = json.loads(quiz_content)
            
            return {
                "questions": quiz_data["questions"],
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generando quiz: {e}")
            return {
                "questions": self._generate_fallback_quiz(text, key_concepts, num_questions),
                "success": False,
                "error": str(e)
            }
    
    async def generate_feedback(self, score: int, total: int, incorrect_questions: List[int], concepts: List[str]) -> str:
        """
        Genera retroalimentación pedagógica personalizada
        """
        if not self.client:
            return self._generate_fallback_feedback(score, total)
        
        try:
            percentage = (score / total) * 100
            
            prompt = f"""
            Genera retroalimentación pedagógica constructiva para un estudiante que obtuvo {score} de {total} preguntas correctas ({percentage:.1f}%).

            PREGUNTAS INCORRECTAS: {incorrect_questions if incorrect_questions else "Ninguna"}
            CONCEPTOS DEL TEMA: {", ".join(concepts[:5])}

            GENERA:
            1. Felicitación o motivación apropiada
            2. Análisis constructivo del desempeño
            3. 2-3 sugerencias específicas de mejora
            4. Recursos o estrategias de estudio recomendadas

            TONO: Alentador, constructivo y educativo
            IDIOMA: Español
            LONGITUD: 100-150 palabras
            """

            response = self.client.chat.completions.create(  # ← CAMBIO AQUÍ
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un tutor educativo empático y constructivo."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generando feedback: {e}")
            return self._generate_fallback_feedback(score, total)
    
    def _generate_fallback_summary(self, text: str) -> str:
        """Genera un resumen básico sin IA"""
        sentences = text.split('.')[:3]
        return f"Resumen generado localmente: {'. '.join(sentences)}."
    
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