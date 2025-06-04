#!/usr/bin/env python3
"""
setup_models.py - Script para configurar y descargar modelos de IA gratuitos

Ejecutar una vez después de la instalación para pre-descargar todos los modelos.
"""

import os
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Verificar que las dependencias estén instaladas"""
    required_packages = [
        'transformers',
        'torch',
        'accelerate',
        'sentence_transformers',
        'huggingface_hub'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} instalado")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} no encontrado")
    
    if missing_packages:
        logger.error(f"Faltan paquetes: {', '.join(missing_packages)}")
        logger.info("Ejecuta: pip install -r requirements.txt")
        return False
    
    return True

def check_gpu():
    """Verificar disponibilidad de GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ GPU disponible: {gpu_name} ({gpu_count} dispositivos)")
            return True
        else:
            logger.info("💻 Solo CPU disponible - los modelos funcionarán más lento")
            return False
    except Exception as e:
        logger.warning(f"Error verificando GPU: {e}")
        return False

def download_models():
    """Descargar y cachear todos los modelos necesarios"""
    logger.info("🚀 Descargando modelos de IA gratuitos...")
    
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSeq2SeqLM,
            T5ForConditionalGeneration, T5Tokenizer,
            pipeline
        )
        
        # Crear directorio de caché
        cache_dir = Path("model_cache")
        cache_dir.mkdir(exist_ok=True)
        
        models_to_download = [
            {
                "name": "Resumidor BART",
                "model": "facebook/bart-large-cnn",
                "type": "pipeline",
                "task": "summarization"
            },
            {
                "name": "Generador T5",
                "model": "google/flan-t5-base", 
                "type": "t5",
                "task": "text2text-generation"
            },
            {
                "name": "Analizador RoBERTa",
                "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "type": "pipeline", 
                "task": "text-classification"
            }
        ]
        
        for model_info in models_to_download:
            logger.info(f"📥 Descargando {model_info['name']}...")
            
            try:
                if model_info["type"] == "pipeline":
                    model = pipeline(
                        model_info["task"],
                        model=model_info["model"],
                        cache_dir=str(cache_dir)
                    )
                elif model_info["type"] == "t5":
                    tokenizer = T5Tokenizer.from_pretrained(
                        model_info["model"],
                        cache_dir=str(cache_dir)
                    )
                    model = T5ForConditionalGeneration.from_pretrained(
                        model_info["model"], 
                        cache_dir=str(cache_dir)
                    )
                
                logger.info(f"✅ {model_info['name']} descargado correctamente")
                
            except Exception as e:
                logger.error(f"❌ Error descargando {model_info['name']}: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error general descargando modelos: {e}")
        return False

def test_models():
    """Probar que todos los modelos funcionen correctamente"""
    logger.info("🧪 Probando modelos...")
    
    try:
        from app.services.ai_service import AIService
        
        # Crear instancia del servicio
        ai_service = AIService()
        
        # Test de resumen
        test_text = "La inteligencia artificial es una rama de la informática que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana."
        
        logger.info("📝 Probando generación de resumen...")
        import asyncio
        
        async def test_async():
            summary_result = await ai_service.generate_summary(test_text, "short")
            if summary_result["success"]:
                logger.info("✅ Resumen generado correctamente")
                logger.info(f"Resumen: {summary_result['summary'][:100]}...")
            else:
                logger.error("❌ Error generando resumen")
                return False
            
            # Test de quiz
            logger.info("❓ Probando generación de quiz...")
            quiz_result = await ai_service.generate_quiz(test_text, ["inteligencia artificial"], 2, "medium")
            if quiz_result["success"] and len(quiz_result["questions"]) > 0:
                logger.info("✅ Quiz generado correctamente")
                logger.info(f"Primera pregunta: {quiz_result['questions'][0]['question']}")
            else:
                logger.error("❌ Error generando quiz")
                return False
            
            return True
        
        result = asyncio.run(test_async())
        return result
        
    except Exception as e:
        logger.error(f"Error probando modelos: {e}")
        return False

def print_system_info():
    """Mostrar información del sistema"""
    import platform
    import psutil
    
    logger.info("💻 Información del sistema:")
    logger.info(f"   OS: {platform.system()} {platform.release()}")
    logger.info(f"   Python: {platform.python_version()}")
    logger.info(f"   CPU: {platform.processor()}")
    logger.info(f"   RAM: {psutil.virtual_memory().total // (1024**3)} GB")
    
    try:
        import torch
        logger.info(f"   PyTorch: {torch.__version__}")
        logger.info(f"   CUDA: {torch.version.cuda if torch.cuda.is_available() else 'No disponible'}")
    except:
        logger.info("   PyTorch: No instalado")

def main():
    """Función principal del script de configuración"""
    logger.info("🎯 Configurando IA Educativa con modelos gratuitos")
    logger.info("=" * 60)
    
    # 1. Mostrar info del sistema
    print_system_info()
    
    # 2. Verificar dependencias
    logger.info("\n📦 Verificando dependencias...")
    if not check_dependencies():
        logger.error("❌ Faltan dependencias. Instalación abortada.")
        sys.exit(1)
    
    # 3. Verificar GPU
    logger.info("\n🔍 Verificando hardware...")
    gpu_available = check_gpu()
    
    # 4. Descargar modelos
    logger.info("\n📥 Descargando modelos...")
    if not download_models():
        logger.error("❌ Error descargando modelos. Instalación abortada.")
        sys.exit(1)
    
    # 5. Probar modelos
    logger.info("\n🧪 Probando funcionalidad...")
    if not test_models():
        logger.error("❌ Error probando modelos. Puede haber problemas.")
        sys.exit(1)
    
    # 6. Finalización exitosa
    logger.info("\n" + "=" * 60)
    logger.info("🎉 ¡Configuración completada exitosamente!")
    logger.info("💡 Consejos para optimizar rendimiento:")
    
    if gpu_available:
        logger.info("   • Tu GPU está disponible - los modelos serán rápidos")
        logger.info("   • Configura AI_USE_GPU=true en tu .env")
    else:
        logger.info("   • Solo tienes CPU - considera usar modelos pequeños")
        logger.info("   • Configura AI_MODEL_SIZE=small en tu .env")
    
    logger.info("\n🚀 Puedes ejecutar el servidor:")
    logger.info("   uvicorn app.main:app --reload")
    
    logger.info("\n📊 Espacio usado por modelos:")
    cache_dir = Path("model_cache")
    if cache_dir.exists():
        size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        logger.info(f"   ~{size // (1024**2)} MB en model_cache/")

if __name__ == "__main__":
    main()