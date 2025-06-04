#!/usr/bin/env python3
"""
test_models.py - Script para probar todos los modelos y funcionalidades

Ejecutar para verificar que todo funciona correctamente.
"""

import asyncio
import time
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Textos de prueba
TEST_TEXTS = {
    "ciencias": """
    La fotosíntesis es el proceso biológico más importante para la vida en la Tierra. 
    Las plantas utilizan la energía solar para convertir dióxido de carbono y agua en glucosa y oxígeno.
    Este proceso ocurre en los cloroplastos, específicamente en los tilacoides donde se encuentra la clorofila.
    La fotosíntesis consta de dos fases principales: las reacciones dependientes de luz y el ciclo de Calvin.
    Durante las reacciones de luz, se produce ATP y NADPH, mientras que en el ciclo de Calvin se fija el CO2.
    """,
    
    "historia": """
    La Revolución Industrial comenzó en Inglaterra a finales del siglo XVIII y transformó completamente la sociedad.
    La invención de la máquina de vapor por James Watt en 1769 revolucionó el transporte y la producción.
    Las fábricas textiles fueron las primeras en adoptar la mecanización, especialmente en Manchester.
    La construcción de ferrocarriles conectó ciudades y facilitó el comercio de materias primas.
    Esta revolución cambió las estructuras sociales, creando una nueva clase trabajadora urbana.
    """,
    
    "tecnologia": """
    La inteligencia artificial ha evolucionado dramáticamente en las últimas décadas.
    Los algoritmos de machine learning permiten a las máquinas aprender patrones de datos.
    Las redes neuronales artificiales se inspiran en el funcionamiento del cerebro humano.
    El deep learning utiliza múltiples capas para procesar información compleja.
    Aplicaciones como el reconocimiento de voz y la visión por computadora son ya realidad.
    """
}

async def test_summary_generation():
    """Probar generación de resúmenes"""
    logger.info("📝 PROBANDO: Generación de resúmenes")
    
    try:
        from app.services.ai_service import AIService
        ai_service = AIService()
        
        results = {}
        
        for topic, text in TEST_TEXTS.items():
            logger.info(f"   Procesando: {topic}")
            
            start_time = time.time()
            result = await ai_service.generate_summary(text, "medium")
            end_time = time.time()
            
            if result["success"]:
                logger.info(f"   ✅ {topic}: {end_time - start_time:.2f}s")
                results[topic] = {
                    "success": True,
                    "time": end_time - start_time,
                    "summary_length": len(result["summary"]),
                    "summary_preview": result["summary"][:100] + "..."
                }
            else:
                logger.error(f"   ❌ {topic}: {result.get('error', 'Error desconocido')}")
                results[topic] = {"success": False, "error": result.get("error")}
        
        return results
        
    except Exception as e:
        logger.error(f"Error en test de resúmenes: {e}")
        return {"error": str(e)}

async def test_quiz_generation():
    """Probar generación de quizzes"""
    logger.info("❓ PROBANDO: Generación de quizzes")
    
    try:
        from app.services.ai_service import AIService
        ai_service = AIService()
        
        results = {}
        
        for topic, text in TEST_TEXTS.items():
            logger.info(f"   Procesando: {topic}")
            
            # Extraer conceptos clave básicos
            concepts = text.split()[:5]  # Primeras 5 palabras como conceptos
            
            start_time = time.time()
            result = await ai_service.generate_quiz(text, concepts, 3, "medium")
            end_time = time.time()
            
            if result["success"] and len(result["questions"]) > 0:
                logger.info(f"   ✅ {topic}: {len(result['questions'])} preguntas en {end_time - start_time:.2f}s")
                results[topic] = {
                    "success": True,
                    "time": end_time - start_time,
                    "question_count": len(result["questions"]),
                    "first_question": result["questions"][0]["question"] if result["questions"] else "N/A"
                }
            else:
                logger.error(f"   ❌ {topic}: {result.get('error', 'Error desconocido')}")
                results[topic] = {"success": False, "error": result.get("error")}
        
        return results
        
    except Exception as e:
        logger.error(f"Error en test de quizzes: {e}")
        return {"error": str(e)}

async def test_feedback_generation():
    """Probar generación de retroalimentación"""
    logger.info("💬 PROBANDO: Generación de feedback")
    
    try:
        from app.services.ai_service import AIService
        ai_service = AIService()
        
        # Casos de prueba con diferentes puntuaciones
        test_cases = [
            {"score": 5, "total": 5, "concepts": ["fotosíntesis", "clorofila"]},
            {"score": 3, "total": 5, "concepts": ["revolución", "industrial"]},
            {"score": 1, "total": 5, "concepts": ["inteligencia", "artificial"]}
        ]
        
        results = {}
        
        for i, case in enumerate(test_cases):
            logger.info(f"   Caso {i+1}: {case['score']}/{case['total']}")
            
            start_time = time.time()
            feedback = await ai_service.generate_feedback(
                case["score"], 
                case["total"], 
                [], 
                case["concepts"]
            )
            end_time = time.time()
            
            logger.info(f"   ✅ Feedback generado en {end_time - start_time:.2f}s")
            results[f"case_{i+1}"] = {
                "score": f"{case['score']}/{case['total']}",
                "time": end_time - start_time,
                "feedback_length": len(feedback),
                "feedback_preview": feedback[:100] + "..."
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error en test de feedback: {e}")
        return {"error": str(e)}

def test_model_loading():
    """Probar carga de modelos"""
    logger.info("🤖 PROBANDO: Carga de modelos")
    
    try:
        start_time = time.time()
        from app.services.ai_service import AIService
        ai_service = AIService()
        end_time = time.time()
        
        logger.info(f"   ✅ Modelos cargados en {end_time - start_time:.2f}s")
        
        # Verificar que los modelos estén cargados
        has_summarizer = hasattr(ai_service, 'summarizer') and ai_service.summarizer is not None
        has_t5 = hasattr(ai_service, 't5_model') and ai_service.t5_model is not None
        has_classifier = hasattr(ai_service, 'classifier') and ai_service.classifier is not None
        
        return {
            "loading_time": end_time - start_time,
            "summarizer_loaded": has_summarizer,
            "t5_loaded": has_t5,
            "classifier_loaded": has_classifier,
            "device": ai_service.device if hasattr(ai_service, 'device') else "unknown"
        }
        
    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")
        return {"error": str(e)}

def check_model_cache():
    """Verificar caché de modelos"""
    logger.info("💾 VERIFICANDO: Caché de modelos")
    
    cache_dir = Path("model_cache")
    if not cache_dir.exists():
        logger.warning("   ⚠️ Directorio model_cache no existe")
        return {"cache_exists": False}
    
    # Calcular tamaño del caché
    total_size = 0
    file_count = 0
    
    for file_path in cache_dir.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
            file_count += 1
    
    size_mb = total_size / (1024 * 1024)
    logger.info(f"   📊 Caché: {file_count} archivos, {size_mb:.1f} MB")
    
    return {
        "cache_exists": True,
        "file_count": file_count,
        "size_mb": round(size_mb, 1),
        "path": str(cache_dir.absolute())
    }

def check_dependencies():
    """Verificar dependencias críticas"""
    logger.info("📦 VERIFICANDO: Dependencias")
    
    dependencies = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "accelerate": "Accelerate",
        "sentence_transformers": "Sentence Transformers"
    }
    
    results = {}
    
    for module, name in dependencies.items():
        try:
            imported_module = __import__(module)
            version = getattr(imported_module, '__version__', 'unknown')
            logger.info(f"   ✅ {name}: {version}")
            results[module] = {"installed": True, "version": version}
        except ImportError:
            logger.error(f"   ❌ {name}: No instalado")
            results[module] = {"installed": False}
    
    # Verificar GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"   🎮 GPU: {gpu_name}")
            results["gpu"] = {"available": True, "name": gpu_name}
        else:
            logger.info("   💻 GPU: No disponible (usando CPU)")
            results["gpu"] = {"available": False}
    except:
        results["gpu"] = {"available": False, "error": "No se pudo verificar"}
    
    return results

async def run_full_test():
    """Ejecutar todos los tests"""
    logger.info("🚀 INICIANDO TESTS COMPLETOS")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. Verificar dependencias
    results["dependencies"] = check_dependencies()
    
    # 2. Verificar caché
    results["model_cache"] = check_model_cache()
    
    # 3. Probar carga de modelos
    results["model_loading"] = test_model_loading()
    
    # 4. Probar resúmenes
    results["summary_generation"] = await test_summary_generation()
    
    # 5. Probar quizzes
    results["quiz_generation"] = await test_quiz_generation()
    
    # 6. Probar feedback
    results["feedback_generation"] = await test_feedback_generation()
    
    return results

def generate_report(results):
    """Generar reporte de resultados"""
    logger.info("\n" + "=" * 60)
    logger.info("📊 REPORTE DE RESULTADOS")
    logger.info("=" * 60)
    
    # Guardar resultados en JSON
    report_file = Path("test_results.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"💾 Reporte completo guardado en: {report_file}")
    
    # Resumen ejecutivo
    total_tests = 0
    passed_tests = 0
    
    for category, data in results.items():
        if isinstance(data, dict):
            if category == "dependencies":
                for dep, info in data.items():
                    if dep != "gpu":
                        total_tests += 1
                        if info.get("installed", False):
                            passed_tests += 1
            elif category in ["summary_generation", "quiz_generation"]:
                if "error" not in data:
                    for topic, info in data.items():
                        total_tests += 1
                        if info.get("success", False):
                            passed_tests += 1
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    logger.info(f"\n🎯 RESUMEN:")
    logger.info(f"   Tests ejecutados: {total_tests}")
    logger.info(f"   Tests exitosos: {passed_tests}")
    logger.info(f"   Tasa de éxito: {success_rate:.1f}%")
    
    if success_rate >= 90:
        logger.info("   🎉 ¡Excelente! Todo está funcionando correctamente")
    elif success_rate >= 70:
        logger.info("   ⚠️ Hay algunos problemas menores")
    else:
        logger.info("   ❌ Se detectaron problemas importantes")
    
    logger.info(f"\n📁 Revisa {report_file} para detalles completos")

async def main():
    """Función principal"""
    start_time = time.time()
    
    try:
        results = await run_full_test()
        generate_report(results)
        
        end_time = time.time()
        logger.info(f"\n⏱️ Tests completados en {end_time - start_time:.2f} segundos")
        
    except Exception as e:
        logger.error(f"❌ Error ejecutando tests: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())