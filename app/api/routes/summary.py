# app/api/routes/summary.py - CORREGIDO
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from typing import List

from app.models.schemas import SummaryRequest
from app.models.response_models import APIResponse, SummaryResponse, VisualizationResponse, ChartData
from app.services.service_manager import service_manager  # ← CAMBIO
from app.utils.helpers import estimate_reading_time

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/generate-summary", response_model=APIResponse)
async def generate_summary(request: SummaryRequest):
    """
    Genera un resumen educativo del texto
    """
    try:
        # ✅ USAR SERVICE_MANAGER (instancias singleton)
        ai_service = service_manager.ai_service
        nlp_service = service_manager.nlp_service
        
        # Generar resumen con IA
        summary_result = await ai_service.generate_summary(request.text, request.length)
        
        # Extraer conceptos clave
        key_concepts = nlp_service.extract_key_concepts(request.text, max_concepts=8)
        
        # Análisis de texto
        text_analysis = nlp_service.analyze_text_complexity(request.text)
        
        # Crear respuesta estructurada
        summary_response = SummaryResponse(
            summary=summary_result["summary"],
            key_concepts=[
                {
                    "concept": concept["concept"],
                    "frequency": concept["frequency"],
                    "relevance": concept["relevance"]
                }
                for concept in key_concepts
            ],
            word_count=text_analysis["word_count"],
            reading_time=text_analysis["reading_time"],
            generated_at=datetime.utcnow()
        )
        
        return APIResponse(
            success=True,
            message="Resumen generado exitosamente",
            data=summary_response.dict()
        )
        
    except Exception as e:
        logger.error(f"Error generando resumen: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error generando el resumen"
        )

@router.post("/generate-visualization", response_model=APIResponse)
async def generate_visualization(request: SummaryRequest):
    """
    Genera visualizaciones de los conceptos del texto
    """
    try:
        # ✅ USAR SERVICE_MANAGER (instancia singleton)
        nlp_service = service_manager.nlp_service
        
        # Extraer conceptos clave
        key_concepts = nlp_service.extract_key_concepts(request.text, max_concepts=10)
        
        # Crear datos para gráfico de barras
        chart_data = [
            ChartData(
                name=concept["concept"],
                value=concept["frequency"],
                category="concepto"
            )
            for concept in key_concepts[:8]
        ]
        
        # Determinar tipo de visualización
        chart_type = "bar"
        title = "Distribución de Conceptos Clave"
        description = f"Frecuencia de aparición de los {len(chart_data)} conceptos más importantes en el texto"
        
        text_lower = request.text.lower()
        if any(word in text_lower for word in ["historia", "cronología", "fecha", "año", "siglo", "época"]):
            chart_type = "timeline"
            title = "Línea de Tiempo Histórica"
            description = "Eventos y fechas importantes identificados en el texto"
        elif any(word in text_lower for word in ["porcentaje", "%", "estadística", "proporción", "parte"]):
            chart_type = "pie"
            title = "Distribución Proporcional"
            description = "Distribución porcentual de conceptos en el contenido"
        
        visualization_response = VisualizationResponse(
            chart_type=chart_type,
            data=chart_data,
            title=title,
            description=description
        )
        
        return APIResponse(
            success=True,
            message="Visualización generada exitosamente",
            data=visualization_response.dict()
        )
        
    except Exception as e:
        logger.error(f"Error generando visualización: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error generando la visualización"
        )