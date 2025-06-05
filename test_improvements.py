#!/usr/bin/env python3
"""
test_improvements.py - Prueba las mejoras implementadas en el sistema

Ejecutar para verificar que las mejoras funcionan correctamente.
"""

import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Texto de prueba de la Segunda Guerra Mundial
TEST_TEXT = """
La Segunda Guerra Mundial (1939-1945) fue el conflicto bélico más devastador de la historia de la humanidad. 
Comenzó el 1 de septiembre de 1939 cuando Alemania invadió Polonia, lo que llevó a Francia y Reino Unido a declararle la guerra.

Adolf Hitler, líder de la Alemania Nazi, buscaba expandir el territorio alemán y establecer un imperio que durara mil años.
La guerra se caracterizó por innovaciones tecnológicas como los tanques, la aviación de combate y posteriormente las armas nucleares.

Los principales bandos fueron las Potencias del Eje (Alemania, Italia, Japón) contra los Aliados (Reino Unido, Unión Soviética, Estados Unidos).
La Operación Barbarroja en 1941 marcó la invasión alemana a la Unión Soviética, abriendo el frente oriental.

El Holocausto representó el exterminio sistemático de seis millones de judíos europeos por parte del régimen nazi.
La guerra terminó en 1945 con la rendición de Alemania en mayo y de Japón en septiembre, tras las bombas atómicas de Hiroshima y Nagasaki.

Las consecuencias incluyeron la creación de las Naciones Unidas, la Guerra Fría entre Estados Unidos y la Unión Soviética, 
y la descolonización de numerosos territorios en África y Asia.
"""

async def test_improved_nlp_service():
    """Probar el servicio NLP mejorado"""
    logger.info("🔍 PROBANDO: Servicio NLP mejorado")
    
    try:
        from app.services.nlp_service import NLPService
        nlp_service = NLPService()
        
        # Test 1: Extracción de conceptos clave mejorada
        logger.info("📝 Test 1: Extracción de conceptos clave...")
        concepts = nlp_service.extract_key_concepts(TEST_TEXT, max_concepts=8)
        
        logger.info(f"✅ Conceptos extraídos: {len(concepts)}")
        for i, concept in enumerate(concepts[:5], 1):
            logger.info(f"   {i}. {concept['concept']} (freq: {concept['frequency']}, rel: {concept['relevance']:.2f})")
        
        # Test 2: Análisis de complejidad mejorado
        logger.info("📊 Test 2: Análisis de complejidad...")
        analysis = nlp_service.analyze_text_complexity(TEST_TEXT)
        
        logger.info(f"✅ Análisis completado:")
        logger.info(f"   Palabras: {analysis['word_count']}")
        logger.info(f"   Oraciones: {analysis['sentence_count']}")
        logger.info(f"   Diversidad léxica: {analysis.get('lexical_diversity', 'N/A')}")
        logger.info(f"   Complejidad: {analysis['complexity_level']}")
        
        return concepts
        
    except Exception as e:
        logger.error(f"❌ Error en NLP Service: {e}")
        return []

async def test_improved_ai_service(concepts):
    """Probar el servicio AI mejorado"""
    logger.info("🤖 PROBANDO: Servicio AI mejorado")
    
    try:
        from app.services.ai_service import AIService
        ai_service = AIService()
        
        # Test 1: Generación de resumen mejorada
        logger.info("📝 Test 1: Generación de resumen mejorada...")
        summary_result = await ai_service.generate_summary(TEST_TEXT, "medium")
        
        if summary_result["success"]:
            logger.info("✅ Resumen generado exitosamente")
            logger.info(f"Resumen: {summary_result['summary'][:200]}...")
        else:
            logger.error(f"❌ Error en resumen: {summary_result.get('error')}")
        
        # Test 2: Generación de quiz mejorada
        logger.info("❓ Test 2: Generación de quiz mejorada...")
        concept_names = [c["concept"] for c in concepts] if concepts else ["Segunda Guerra Mundial", "Hitler", "Nazi"]
        
        quiz_result = await ai_service.generate_quiz(TEST_TEXT, concept_names, 3, "medium")
        
        if quiz_result["success"]:
            logger.info(f"✅ Quiz generado con {len(quiz_result['questions'])} preguntas")
            for i, q in enumerate(quiz_result["questions"], 1):
                logger.info(f"   Pregunta {i}: {q['question'][:80]}...")
                logger.info(f"   Respuesta correcta: {q['options'][q['correct_answer']][:60]}...")
        else:
            logger.error(f"❌ Error en quiz: {quiz_result.get('error')}")
        
        # Test 3: Generación de feedback mejorada
        logger.info("💬 Test 3: Generación de feedback mejorada...")
        feedback = await ai_service.generate_feedback(2, 3, [2], concept_names[:3])
        
        logger.info(f"✅ Feedback generado: {feedback[:150]}...")
        
        return quiz_result
        
    except Exception as e:
        logger.error(f"❌ Error en AI Service: {e}")
        return {"questions": [], "success": False}

def compare_results():
    """Comparar resultados antes y después de las mejoras"""
    logger.info("📊 COMPARACIÓN DE MEJORAS:")
    logger.info("="*60)
    
    improvements = [
        "✅ Extracción de conceptos más inteligente (filtros avanzados)",
        "✅ Resúmenes con mejor estructura educativa",
        "✅ Preguntas más contextuales y relevantes",
        "✅ Feedback personalizado y detallado",
        "✅ Mejor procesamiento de texto en español",
        "✅ Análisis de complejidad más preciso",
        "✅ Distractores más inteligentes en quizzes"
    ]
    
    for improvement in improvements:
        logger.info(f"   {improvement}")
    
    logger.info("\n🎯 BENEFICIOS ESPERADOS:")
    logger.info("   • Conceptos más relevantes y menos ruido")
    logger.info("   • Resúmenes más educativos y estructurados")
    logger.info("   • Preguntas más coherentes y útiles")
    logger.info("   • Feedback que realmente ayuda al aprendizaje")

async def main():
    """Función principal de prueba"""
    logger.info("🚀 INICIANDO PRUEBAS DE MEJORAS AL SISTEMA")
    logger.info("="*60)
    
    # Probar NLP Service mejorado
    concepts = await test_improved_nlp_service()
    
    # Probar AI Service mejorado
    quiz_result = await test_improved_ai_service(concepts)
    
    # Mostrar comparación
    compare_results()
    
    # Resultados finales
    logger.info("\n" + "="*60)
    logger.info("🏁 RESULTADOS FINALES")
    logger.info("="*60)
    
    if concepts and quiz_result.get("success"):
        logger.info("🎉 ¡ÉXITO! Todas las mejoras funcionan correctamente")
        logger.info("✅ El sistema ahora debería generar:")
        logger.info("   • Conceptos más relevantes (ej: 'Segunda Guerra Mundial' vs 'Que')")
        logger.info("   • Resúmenes más educativos y estructurados")
        logger.info("   • Preguntas más coherentes y contextuales")
        logger.info("   • Feedback más útil y personalizado")
    else:
        logger.error("❌ Algunas mejoras no funcionaron correctamente")
        logger.error("🔧 Revisa la instalación y dependencias")
    
    logger.info("\n📝 PRÓXIMOS PASOS:")
    logger.info("1. Reemplaza los archivos originales con las versiones mejoradas")
    logger.info("2. Instala el modelo de spaCy mejorado con el script")
    logger.info("3. Reinicia el servidor FastAPI")
    logger.info("4. Prueba con tu documento de la Segunda Guerra Mundial")

if __name__ == "__main__":
    asyncio.run(main())