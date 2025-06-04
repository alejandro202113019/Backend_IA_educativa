# app/api/routes/quiz.py - CORREGIDO
import logging
from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime

from app.models.schemas import QuizRequest, QuizSubmission
from app.models.response_models import APIResponse, QuizResponse, QuizQuestion, QuizResult
from app.services.service_manager import service_manager  # ← CAMBIO

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/generate-quiz", response_model=APIResponse)
async def generate_quiz(request: QuizRequest):
    """
    Genera un quiz educativo basado en el texto
    """
    try:
        # ✅ USAR SERVICE_MANAGER (instancias singleton)
        ai_service = service_manager.ai_service
        nlp_service = service_manager.nlp_service
        quiz_manager = service_manager.quiz_manager
        
        # Extraer conceptos clave del texto
        key_concepts = nlp_service.extract_key_concepts(request.text, max_concepts=10)
        concept_names = [concept["concept"] for concept in key_concepts]
        
        # Generar preguntas con IA
        quiz_result = await ai_service.generate_quiz(
            text=request.text,
            key_concepts=concept_names,
            num_questions=request.num_questions,
            difficulty=request.difficulty
        )
        
        if not quiz_result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Error generando quiz: {quiz_result.get('error', 'Error desconocido')}"
            )
        
        # Convertir a objetos QuizQuestion
        questions = [
            QuizQuestion(
                id=q["id"],
                question=q["question"],
                options=q["options"],
                correct_answer=q["correct_answer"],
                explanation=q["explanation"],
                difficulty=q["difficulty"]
            )
            for q in quiz_result["questions"]
        ]
        
        # Crear sesión de quiz
        session_id = quiz_manager.create_quiz_session(questions)
        
        # Estimar tiempo del quiz (2-3 minutos por pregunta)
        estimated_time = len(questions) * 2
        
        quiz_response = QuizResponse(
            questions=questions,
            session_id=session_id,
            total_questions=len(questions),
            estimated_time=estimated_time
        )
        
        return APIResponse(
            success=True,
            message="Quiz generado exitosamente",
            data=quiz_response.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generando quiz: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno generando el quiz"
        )

@router.post("/submit-quiz", response_model=APIResponse)
async def submit_quiz(submission: QuizSubmission):
    """
    Procesa las respuestas del quiz y genera retroalimentación
    """
    try:
        # ✅ USAR SERVICE_MANAGER (instancias singleton)
        ai_service = service_manager.ai_service
        quiz_manager = service_manager.quiz_manager
        
        # Convertir respuestas al formato esperado
        answers = [
            {
                "question_id": answer.question_id,
                "selected_answer": answer.selected_answer
            }
            for answer in submission.answers
        ]
        
        # Procesar respuestas
        result = quiz_manager.submit_quiz_answers(submission.session_id, answers)
        
        # Obtener sesión para contexto adicional
        session = quiz_manager.get_session(submission.session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Sesión de quiz no encontrada"
            )
        
        # Extraer conceptos de las preguntas para feedback personalizado
        questions = session["questions"]
        concepts = []
        incorrect_questions = []
        
        for i, answer in enumerate(answers):
            question_id = answer["question_id"]
            selected = answer["selected_answer"]
            
            if question_id <= len(questions):
                question = questions[question_id - 1]
                # Extraer conceptos principales de cada pregunta
                question_text = question["question"]
                words = question_text.split()
                concepts.extend([word for word in words if len(word) > 4])
                
                # Identificar preguntas incorrectas
                if selected != question["correct_answer"]:
                    incorrect_questions.append(question_id)
        
        # Generar feedback personalizado con IA
        try:
            ai_feedback = await ai_service.generate_feedback(
                score=result.score,
                total=result.total_questions,
                incorrect_questions=incorrect_questions,
                concepts=list(set(concepts))[:5]  # Top 5 conceptos únicos
            )
            result.feedback = ai_feedback
        except Exception as e:
            logger.warning(f"Error generando feedback con IA: {e}")
            # El feedback por defecto ya está en result
        
        return APIResponse(
            success=True,
            message="Quiz evaluado exitosamente",
            data=result.dict()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error procesando quiz: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error procesando las respuestas del quiz"
        )

@router.get("/quiz-session/{session_id}", response_model=APIResponse)
async def get_quiz_session(session_id: str):
    """
    Obtiene información de una sesión de quiz
    """
    try:
        # ✅ USAR SERVICE_MANAGER (instancia singleton)
        quiz_manager = service_manager.quiz_manager
        
        session = quiz_manager.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Sesión no encontrada o expirada"
            )
        
        return APIResponse(
            success=True,
            message="Sesión encontrada",
            data={
                "session_id": session_id,
                "created_at": session["created_at"].isoformat(),
                "completed": session["completed"],
                "total_questions": len(session["questions"]),
                "score": session.get("score")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo sesión: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error obteniendo información de la sesión"
        )