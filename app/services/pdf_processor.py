# app/services/pdf_processor.py
import PyPDF2
import pytesseract
from PIL import Image
import io
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        # Configurar Tesseract si es necesario
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
        pass
    
    async def extract_text_from_pdf(self, file_content: bytes) -> Dict[str, Any]:
        """
        Extrae texto de un archivo PDF
        """
        try:
            text_content = ""
            metadata = {}
            
            # Crear objeto PDF desde bytes
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extraer metadata
            metadata = {
                "num_pages": len(pdf_reader.pages),
                "title": pdf_reader.metadata.get('/Title', '') if pdf_reader.metadata else '',
                "author": pdf_reader.metadata.get('/Author', '') if pdf_reader.metadata else '',
                "creator": pdf_reader.metadata.get('/Creator', '') if pdf_reader.metadata else ''
            }
            
            # Extraer texto de cada página
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content += f"\n--- Página {page_num + 1} ---\n"
                        text_content += page_text
                    else:
                        # Si no hay texto, intentar OCR en imágenes
                        logger.info(f"Intentando OCR en página {page_num + 1}")
                        # Aquí iría la lógica de OCR si es necesaria
                        
                except Exception as e:
                    logger.error(f"Error procesando página {page_num + 1}: {e}")
                    continue
            
            return {
                "text": text_content.strip(),
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error extrayendo texto del PDF: {e}")
            return {
                "text": "",
                "metadata": {},
                "success": False,
                "error": str(e)
            }
    
    async def extract_text_from_txt(self, file_content: bytes, encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Extrae texto de un archivo de texto plano
        """
        try:
            # Intentar diferentes codificaciones
            encodings = [encoding, 'utf-8', 'latin-1', 'cp1252']
            
            for enc in encodings:
                try:
                    text_content = file_content.decode(enc)
                    return {
                        "text": text_content,
                        "metadata": {"encoding": enc, "size": len(file_content)},
                        "success": True
                    }
                except UnicodeDecodeError:
                    continue
            
            # Si no funciona ninguna codificación
            return {
                "text": "",
                "metadata": {},
                "success": False,
                "error": "No se pudo decodificar el archivo con ninguna codificación"
            }
            
        except Exception as e:
            logger.error(f"Error extrayendo texto del archivo: {e}")
            return {
                "text": "",
                "metadata": {},
                "success": False,
                "error": str(e)
            }
