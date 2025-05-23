# app/models/schemas.py CORREGIDO
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class TextInput(BaseModel):
    content: str = Field(..., min_length=10, max_length=50000, description="Contenido del texto a procesar")
    title: Optional[str] = Field(None, max_length=200, description="Título opcional del contenido")

class SummaryRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Texto para generar resumen")
    length: Optional[str] = Field("medium", pattern="^(short|medium|long)$", description="Longitud del resumen")  # ← CAMBIO

class QuizRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Texto base para generar quiz")
    num_questions: Optional[int] = Field(5, ge=3, le=15, description="Número de preguntas")
    difficulty: Optional[str] = Field("medium", pattern="^(easy|medium|hard)$", description="Dificultad del quiz")  # ← CAMBIO

class QuizAnswer(BaseModel):
    question_id: int
    selected_answer: int
    
class QuizSubmission(BaseModel):
    answers: List[QuizAnswer]
    session_id: str