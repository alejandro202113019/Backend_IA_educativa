#!/usr/bin/env python3
"""
test_singleton.py - Verificar que el patrón singleton funciona correctamente
"""

import asyncio
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_singleton_behavior():
    """Verificar que los servicios son realmente singleton"""
    logger.info("🧪 PROBANDO: Comportamiento Singleton")
    
    # Primera importación
    logger.info("📥 Primera importación del ServiceManager...")
    start_time = time.time()
    from app.services.service_manager import service_manager
    
    # Primera llamada (debería cargar los modelos)
    logger.info("🤖 Primera llamada a ai_service...")
    ai_service_1 = service_manager.ai_service
    first_load_time = time.time() - start_time
    logger.info(f"⏱️ Tiempo de primera carga: {first_load_time:.2f}s")
    
    # Segunda llamada (debería ser instantánea)
    logger.info("🔄 Segunda llamada a ai_service...")
    start_time = time.time()
    ai_service_2 = service_manager.ai_service
    second_load_time = time.time() - start_time
    logger.info(f"⏱️ Tiempo de segunda carga: {second_load_time:.4f}s")
    
    # Verificar que son la misma instancia
    are_same_instance = ai_service_1 is ai_service_2
    logger.info(f"🔗 ¿Misma instancia? {are_same_instance}")
    
    # Verificar IDs de objeto
    logger.info(f"🆔 ID primera instancia: {id(ai_service_1)}")
    logger.info(f"🆔 ID segunda instancia: {id(ai_service_2)}")
    
    # Crear nuevo ServiceManager (debería usar el mismo singleton)
    logger.info("🆕 Creando nuevo ServiceManager...")
    from app.services.service_manager import ServiceManager
    new_manager = ServiceManager()
    
    start_time = time.time()
    ai_service_3 = new_manager.ai_service
    third_load_time = time.time() - start_time
    logger.info(f"⏱️ Tiempo con nuevo manager: {third_load_time:.4f}s")
    
    # Verificar que todas son la misma instancia
    all_same = (ai_service_1 is ai_service_2 is ai_service_3)
    logger.info(f"🔗 ¿Todas la misma instancia? {all_same}")
    
    # Verificar estado del manager
    status = service_manager.get_status()
    logger.info(f"📊 Estado de servicios: {status}")
    
    # Resultados
    logger.info("\n" + "="*50)
    logger.info("📋 RESULTADOS DEL TEST SINGLETON")
    logger.info("="*50)
    
    if all_same and second_load_time < 0.1 and third_load_time < 0.1:
        logger.info("✅ ÉXITO: Patrón singleton funcionando correctamente")
        logger.info("✅ Los modelos se cargan una sola vez")
        logger.info("✅ Las instancias subsecuentes son instantáneas")
    else:
        logger.error("❌ FALLO: Patrón singleton no está funcionando")
        logger.error(f"❌ Tiempo segunda carga: {second_load_time:.4f}s (debería ser <0.1s)")
        logger.error(f"❌ Tiempo tercera carga: {third_load_time:.4f}s (debería ser <0.1s)")
        logger.error(f"❌ Misma instancia: {all_same} (debería ser True)")
    
    return {
        "first_load_time": first_load_time,
        "second_load_time": second_load_time,
        "third_load_time": third_load_time,
        "all_same_instance": all_same,
        "singleton_working": all_same and second_load_time < 0.1
    }

async def test_multiple_requests_simulation():
    """Simular múltiples requests como los del frontend"""
    logger.info("\n🌐 SIMULANDO: Múltiples requests como frontend")
    
    from app.services.service_manager import service_manager
    
    # Simular 5 requests concurrentes
    tasks = []
    for i in range(5):
        tasks.append(simulate_request(i+1))
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    logger.info(f"⏱️ Tiempo total para 5 requests: {total_time:.2f}s")
    logger.info(f"⚡ Tiempo promedio por request: {total_time/5:.2f}s")
    
    # Verificar que todos usaron la misma instancia
    ai_service_ids = [result["ai_service_id"] for result in results]
    all_same_id = len(set(ai_service_ids)) == 1
    
    logger.info(f"🔗 ¿Todos usaron la misma instancia de AI? {all_same_id}")
    
    if total_time < 10 and all_same_id:
        logger.info("✅ ÉXITO: Requests múltiples son eficientes")
    else:
        logger.error("❌ FALLO: Requests múltiples son lentos")
    
    return {
        "total_time": total_time,
        "avg_time_per_request": total_time/5,
        "all_same_ai_instance": all_same_id,
        "efficient": total_time < 10
    }

async def simulate_request(request_id: int):
    """Simular un request individual"""
    logger.info(f"📨 Request {request_id}: Iniciando...")
    
    start_time = time.time()
    
    # Simular las operaciones típicas de un request
    from app.services.service_manager import service_manager
    
    # 1. Obtener servicios (debería ser instantáneo)
    ai_service = service_manager.ai_service
    nlp_service = service_manager.nlp_service
    
    # 2. Simular análisis de texto
    test_text = f"Este es un texto de prueba para el request {request_id}"
    concepts = nlp_service.extract_key_concepts(test_text, max_concepts=3)
    
    # 3. Simular generación de resumen
    summary = await ai_service.generate_summary(test_text, "short")
    
    end_time = time.time()
    request_time = end_time - start_time
    
    logger.info(f"📨 Request {request_id}: Completado en {request_time:.2f}s")
    
    return {
        "request_id": request_id,
        "time": request_time,
        "ai_service_id": id(ai_service),
        "success": summary.get("success", False)
    }

async def main():
    """Función principal del test"""
    logger.info("🚀 INICIANDO TESTS DE SINGLETON")
    logger.info("="*60)
    
    # Test 1: Verificar comportamiento singleton
    singleton_results = await test_singleton_behavior()
    
    # Test 2: Simular múltiples requests
    requests_results = await test_multiple_requests_simulation()
    
    # Resumen final
    logger.info("\n" + "="*60)
    logger.info("🏁 RESUMEN FINAL")
    logger.info("="*60)
    
    if singleton_results["singleton_working"] and requests_results["efficient"]:
        logger.info("🎉 ¡ÉXITO TOTAL! El sistema está optimizado:")
        logger.info("   ✅ Patrón singleton funcionando")
        logger.info("   ✅ Carga de modelos única")
        logger.info("   ✅ Requests rápidos y eficientes")
        logger.info("   ✅ Listo para producción")
    else:
        logger.error("❌ PROBLEMAS DETECTADOS:")
        if not singleton_results["singleton_working"]:
            logger.error("   ❌ Patrón singleton no funciona")
        if not requests_results["efficient"]:
            logger.error("   ❌ Requests son lentos")
    
    return {
        "singleton": singleton_results,
        "requests": requests_results
    }

if __name__ == "__main__":
    asyncio.run(main())