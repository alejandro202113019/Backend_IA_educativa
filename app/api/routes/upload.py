# app/api/routes/upload.py - CORREGIDO
import os
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional

from app.models.schemas import TextInput
from app.models.response_models import APIResponse
from app.services.pdf_processor import PDFProcessor
from app.services.service_manager import service_manager  # ← CAMBIO
from app.utils.helpers import save_uploaded_file, validate_file_upload, clean_text
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Instanciar solo el processor (no los servicios de IA)
pdf_processor = PDFProcessor()

@router.post("/upload-file", response_model=APIResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None)
):
    """
    Endpoint para subir archivos PDF o texto
    """
    try:
        # Validar archivo
        validate_file_upload(file)
        
        # Verificar tamaño
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Archivo muy grande. Tamaño máximo: {settings.MAX_FILE_SIZE // 1024 // 1024}MB"
            )
        
        # Resetear puntero del archivo
        await file.seek(0)
        
        # Procesar según tipo de archivo
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension == '.pdf':
            result = await pdf_processor.extract_text_from_pdf(content)
        elif file_extension in ['.txt', '.text']:
            result = await pdf_processor.extract_text_from_txt(content)
        else:
            raise HTTPException(
                status_code=400,
                detail="Tipo de archivo no soportado"
            )
        
        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Error procesando archivo: {result.get('error', 'Error desconocido')}"
            )
        
        # Limpiar y analizar texto
        clean_content = clean_text(result["text"])
        
        if len(clean_content) < 50:
            raise HTTPException(
                status_code=400,
                detail="El archivo no contiene suficiente texto para procesar"
            )
        
        # ✅ USAR SERVICE_MANAGER (instancia singleton)
        nlp_service = service_manager.nlp_service
        
        # Extraer conceptos clave
        key_concepts = nlp_service.extract_key_concepts(clean_content)
        text_analysis = nlp_service.analyze_text_complexity(clean_content)
        
        # Programar limpieza del archivo temporal
        file_path = await save_uploaded_file(file, settings.TEMP_FOLDER)
        background_tasks.add_task(os.remove, file_path)
        
        return APIResponse(
            success=True,
            message="Archivo procesado exitosamente",
            data={
                "text": clean_content,
                "title": title or file.filename,
                "metadata": result["metadata"],
                "key_concepts": key_concepts,
                "analysis": text_analysis,
                "file_info": {
                    "filename": file.filename,
                    "size": len(content),
                    "type": file_extension
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en upload_file: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor procesando el archivo"
        )

@router.post("/upload-text", response_model=APIResponse)
async def upload_text(text_input: TextInput):
    """
    Endpoint para procesar texto directo
    """
    try:
        # Limpiar texto
        clean_content = clean_text(text_input.content)
        
        # ✅ USAR SERVICE_MANAGER (instancia singleton)
        nlp_service = service_manager.nlp_service
        
        # Extraer conceptos clave
        key_concepts = nlp_service.extract_key_concepts(clean_content)
        text_analysis = nlp_service.analyze_text_complexity(clean_content)
        
        return APIResponse(
            success=True,
            message="Texto procesado exitosamente",
            data={
                "text": clean_content,
                "title": text_input.title or "Texto ingresado manualmente",
                "key_concepts": key_concepts,
                "analysis": text_analysis,
                "metadata": {
                    "source": "direct_input",
                    "length": len(clean_content)
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error en upload_text: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error procesando el texto"
        )