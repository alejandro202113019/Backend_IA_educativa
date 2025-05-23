# app/utils/helpers.py CORREGIDO
import os
import uuid
import aiofiles
from typing import Optional, List
from fastapi import UploadFile, HTTPException
from app.core.config import settings

async def save_uploaded_file(file: UploadFile, folder: str = "uploads") -> str:
    """
    Guarda un archivo subido y retorna la ruta
    """
    # Crear directorio si no existe
    os.makedirs(folder, exist_ok=True)
    
    # Generar nombre único
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(folder, unique_filename)
    
    # Guardar archivo
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return file_path

def validate_file_upload(file: UploadFile) -> bool:
    """
    Valida que el archivo sea válido para subir
    """
    # Verificar extensión
    file_extension = os.path.splitext(file.filename)[1].lower().lstrip('.')
    if file_extension not in settings.get_allowed_extensions():  # ← USAR MÉTODO
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no permitido. Extensiones permitidas: {', '.join(settings.get_allowed_extensions())}"  # ← USAR MÉTODO
        )
    
    return True

def clean_text(text: str) -> str:
    """
    Limpia y normaliza texto
    """
    import re
    
    # Eliminar caracteres especiales excesivos
    text = re.sub(r'\s+', ' ', text)  # Múltiples espacios -> uno solo
    text = re.sub(r'\n+', '\n', text)  # Múltiples saltos -> uno solo
    text = text.strip()
    
    return text

def estimate_reading_time(text: str, wpm: int = 200) -> int:
    """
    Estima el tiempo de lectura en minutos
    """
    word_count = len(text.split())
    return max(1, word_count // wpm)