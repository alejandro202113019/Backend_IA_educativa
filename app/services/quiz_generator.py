
# app/services/quiz_generator.py
import uuid
import json
from typing import Dict, List, Any
from datetime import datetime, timedelta
from app.models.response_models import QuizQuestion, QuizResult

class QuizManager:
    def __init__(self):
        # En producción, esto debería ser una base de datos
        self.active_sessions: Dict[str, Dict] = {}
        self.session_timeout = timedelta(hours=2)
    
    def create_quiz_session(self, questions: List[QuizQuestion]) -> str:
        """
        Crea una nueva sesión de quiz
        """
        session_id = str(uuid.uuid4())
        
        self.active_sessions[session_id] = {
            "questions": [q.dict() for q in questions],
            "created_at": datetime.utcnow(),
            "completed": False,
            "answers": {},
            "score": None
        }
        
        return session_id
    
    def submit_quiz_answers(self, session_id: str, answers: List[Dict[str, Any]]) -> QuizResult:
        """
        Procesa las respuestas del quiz y calcula el resultado
        """
        if session_id not in self.active_sessions:
            raise ValueError("Sesión de quiz no encontrada o expirada")
        
        session = self.active_sessions[session_id]
        questions = session["questions"]
        
        # Verificar que no esté ya completado
        if session["completed"]:
            raise ValueError("Este quiz ya fue completado")
        
        # Calcular puntuación
        correct_answers = []
        explanations = []
        score = 0
        
        for answer in answers:
            question_id = answer["question_id"]
            selected = answer["selected_answer"]
            
            if question_id <= len(questions):
                question = questions[question_id - 1]
                is_correct = selected == question["correct_answer"]
                
                if is_correct:
                    score += 1
                    correct_answers.append(question_id)
                
                explanations.append(question["explanation"])
        
        # Marcar sesión como completada
        session["completed"] = True
        session["score"] = score
        session["answers"] = {ans["question_id"]: ans["selected_answer"] for ans in answers}
        
        # Generar resultado
        total_questions = len(questions)
        percentage = (score / total_questions) * 100 if total_questions > 0 else 0
        
        # Generar feedback básico
        feedback = self._generate_basic_feedback(score, total_questions, percentage)
        
        # Sugerencias de mejora
        improvement_suggestions = self._generate_improvement_suggestions(score, total_questions, correct_answers)
        
        return QuizResult(
            score=score,
            total_questions=total_questions,
            percentage=round(percentage, 1),
            correct_answers=correct_answers,
            explanations=explanations,
            feedback=feedback,
            improvement_suggestions=improvement_suggestions
        )
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Obtiene información de una sesión
        """
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            # Verificar si la sesión ha expirado
            if datetime.utcnow() - session["created_at"] > self.session_timeout:
                del self.active_sessions[session_id]
                return None
            return session
        return None
    
    def cleanup_expired_sessions(self):
        """
        Limpia sesiones expiradas (debería ejecutarse periódicamente)
        """
        current_time = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if current_time - session["created_at"] > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
    
    def _generate_basic_feedback(self, score: int, total: int, percentage: float) -> str:
        """
        Genera feedback básico basado en la puntuación
        """
        if percentage >= 90:
            return "¡Excelente! Has demostrado un dominio sobresaliente del tema. ¡Felicitaciones!"
        elif percentage >= 80:
            return "¡Muy bien! Tienes un buen entendimiento del tema. Solo algunos puntos por pulir."
        elif percentage >= 70:
            return "Bien hecho. Tienes una comprensión sólida, pero hay espacio para mejorar."
        elif percentage >= 60:
            return "Buen intento. Has captado los conceptos básicos, pero necesitas reforzar algunos temas."
        else:
            return "Te recomiendo revisar el material nuevamente. ¡No te desanimes, el aprendizaje es un proceso!"
    
    def _generate_improvement_suggestions(self, score: int, total: int, correct_answers: List[int]) -> List[str]:
        """
        Genera sugerencias de mejora basadas en el desempeño
        """
        suggestions = []
        percentage = (score / total) * 100 if total > 0 else 0
        
        if percentage < 70:
            suggestions.extend([
                "Revisa el material de estudio nuevamente, enfocándote en los conceptos principales",
                "Toma notas de los puntos clave mientras estudias",
                "Intenta explicar los conceptos con tus propias palabras"
            ])
        elif percentage < 85:
            suggestions.extend([
                "Repasa las preguntas que respondiste incorrectamente",
                "Busca ejemplos adicionales de los conceptos más difíciles",
                "Practica con ejercicios similares"
            ])
        else:
            suggestions.extend([
                "¡Excelente trabajo! Continúa practicando para mantener este nivel",
                "Puedes explorar temas más avanzados relacionados",
                "Comparte tu conocimiento ayudando a otros estudiantes"
            ])
        
        return suggestions
