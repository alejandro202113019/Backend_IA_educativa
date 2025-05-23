# app/models/response_models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class KeyConcept(BaseModel):
    concept: str
    frequency: int
    relevance: float

class SummaryResponse(BaseModel):
    summary: str
    key_concepts: List[KeyConcept]
    word_count: int
    reading_time: int  # en minutos
    generated_at: datetime
    
class ChartData(BaseModel):
    name: str
    value: float
    category: Optional[str] = None

class VisualizationResponse(BaseModel):
    chart_type: str  # bar, line, pie, timeline
    data: List[ChartData]
    title: str
    description: str

class QuizQuestion(BaseModel):
    id: int
    question: str
    options: List[str]
    correct_answer: int
    explanation: str
    difficulty: str

class QuizResponse(BaseModel):
    questions: List[QuizQuestion]
    session_id: str
    total_questions: int
    estimated_time: int  # en minutos

class QuizResult(BaseModel):
    score: int
    total_questions: int
    percentage: float
    correct_answers: List[int]
    explanations: List[str]
    feedback: str
    improvement_suggestions: List[str]

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None
