# diagnose_system.py - Script de diagnóstico completo
"""
Script para diagnosticar todos los problemas del sistema
"""

import asyncio
import os
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_files_structure():
    """Verifica que todos los archivos estén en su lugar"""
    print("🔍 VERIFICANDO ESTRUCTURA DE ARCHIVOS")
    print("=" * 50)
    
    required_files = {
        "app/services/ai_service.py": "Servicio base de IA",
        "app/services/enhanced_ai_service.py": "Servicio mejorado con fine-tuning",
        "app/services/service_manager.py": "Gestor de servicios",
        "models/fine_tuned/model_config.json": "Configuración de modelos",
        "training/lora_trainer.py": "Trainer de LoRA",
        "training/data_preparation.py": "Preparación de datos"
    }
    
    missing_files = []
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"   ✅ {file_path} ({size:.1f} KB) - {description}")
        else:
            print(f"   ❌ {file_path} - FALTANTE - {description}")
            missing_files.append(file_path)
    
    return missing_files

def check_model_config():
    """Verifica la configuración de modelos fine-tuned"""
    print("\n📋 VERIFICANDO CONFIGURACIÓN DE MODELOS")
    print("=" * 50)
    
    config_path = "models/fine_tuned/model_config.json"
    
    if not os.path.exists(config_path):
        print("   ❌ model_config.json NO ENCONTRADO")
        print("   💡 Necesitas ejecutar: python main_training.py")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("   ✅ model_config.json encontrado")
        
        # Verificar modelos
        models = config.get("models", {})
        for model_name, model_info in models.items():
            lora_path = model_info.get("lora_path")
            if os.path.exists(lora_path):
                print(f"   ✅ {model_name}: {lora_path}")
            else:
                print(f"   ❌ {model_name}: {lora_path} - NO ENCONTRADO")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error leyendo configuración: {e}")
        return False

async def test_basic_ai_service():
    """Prueba el servicio de IA básico"""
    print("\n🤖 PROBANDO SERVICIO DE IA BÁSICO")
    print("=" * 50)
    
    try:
        from app.services.ai_service import AIService
        
        ai_service = AIService()
        print("   ✅ AIService inicializado correctamente")
        
        # Test resumen
        test_text = "La inteligencia artificial es una tecnología que permite a las máquinas simular la inteligencia humana."
        
        print("\n   📝 Probando resumen básico...")
        summary_result = await ai_service.generate_summary(test_text, "medium")
        
        if summary_result["success"]:
            print("   ✅ Resumen generado exitosamente")
            print(f"   📄 Resumen: {summary_result['summary'][:100]}...")
        else:
            print(f"   ❌ Error en resumen: {summary_result.get('error', 'Error desconocido')}")
        
        # Test quiz
        print("\n   ❓ Probando generación de quiz...")
        concepts = ["inteligencia artificial", "tecnología"]
        quiz_result = await ai_service.generate_quiz(test_text, concepts, 3, "medium")
        
        if quiz_result["success"]:
            print("   ✅ Quiz generado exitosamente")
            print(f"   📊 Preguntas: {len(quiz_result['questions'])}")
            
            # Mostrar primera pregunta
            if quiz_result['questions']:
                q1 = quiz_result['questions'][0]
                print(f"   📝 Pregunta 1: {q1['question']}")
                print(f"   📋 Opciones: {len(q1['options'])}")
        else:
            print(f"   ❌ Error en quiz: {quiz_result.get('error', 'Error desconocido')}")
        
        # Test feedback
        print("\n   💭 Probando feedback...")
        feedback = await ai_service.generate_feedback(7, 10, [2, 5, 8], concepts)
        print("   ✅ Feedback generado")
        print(f"   📝 Feedback: {feedback[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en AIService: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_enhanced_ai_service():
    """Prueba el servicio de IA mejorado"""
    print("\n🚀 PROBANDO SERVICIO DE IA MEJORADO")
    print("=" * 50)
    
    try:
        from app.services.enhanced_ai_service import EnhancedAIService
        
        enhanced_ai = EnhancedAIService()
        print("   ✅ EnhancedAIService inicializado correctamente")
        
        # Verificar estado de modelos
        status = enhanced_ai.get_model_status()
        print(f"\n   📊 Estado de modelos fine-tuned:")
        for model, loaded in status['fine_tuned_models'].items():
            emoji = "✅" if loaded else "❌"
            print(f"      {emoji} {model}: {'Cargado' if loaded else 'No cargado'}")
        
        # Test resumen mejorado
        test_text = "La inteligencia artificial es una tecnología que permite a las máquinas simular la inteligencia humana mediante algoritmos complejos."
        
        print("\n   📝 Probando resumen mejorado...")
        summary_result = await enhanced_ai.generate_summary(test_text, "medium")
        
        if summary_result["success"]:
            model_used = summary_result.get("model_used", "base_model")
            print(f"   ✅ Resumen generado con: {model_used}")
            print(f"   📄 Resumen: {summary_result['summary'][:150]}...")
        else:
            print(f"   ❌ Error en resumen: {summary_result.get('error', 'Error desconocido')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en EnhancedAIService: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_endpoints():
    """Prueba los endpoints de la API"""
    print("\n🌐 PROBANDO ENDPOINTS DE API")
    print("=" * 50)
    
    try:
        import requests
        base_url = "http://localhost:8000"
        
        # Test health endpoint
        print("   🏥 Probando /health...")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ Health endpoint funciona")
            else:
                print(f"   ❌ Health endpoint error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("   ⚠️ Servidor no está corriendo")
            print("   💡 Ejecuta: uvicorn app.main:app --reload")
            return False
        except Exception as e:
            print(f"   ❌ Error en health: {e}")
        
        # Test summary endpoint
        print("\n   📝 Probando /summary/generate-summary...")
        try:
            payload = {
                "text": "La inteligencia artificial es una tecnología revolucionaria.",
                "length": "medium"
            }
            response = requests.post(f"{base_url}/api/v1/summary/generate-summary", json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    print("   ✅ Summary endpoint funciona")
                    summary = data.get("data", {}).get("summary", "")
                    print(f"   📄 Resumen: {summary[:100]}...")
                else:
                    print(f"   ❌ Summary endpoint falló: {data.get('message', 'Error desconocido')}")
            else:
                print(f"   ❌ Summary endpoint error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error en summary: {e}")
        
        return True
        
    except ImportError:
        print("   ⚠️ requests no instalado - no se pueden probar endpoints")
        print("   💡 Instala con: pip install requests")
        return False

def check_dependencies():
    """Verifica dependencias críticas"""
    print("\n📦 VERIFICANDO DEPENDENCIAS")
    print("=" * 50)
    
    critical_deps = [
        ("torch", "PyTorch para deep learning"),
        ("transformers", "Modelos de Hugging Face"),
        ("peft", "Parameter-Efficient Fine-Tuning"),
        ("datasets", "Manejo de datasets"),
        ("fastapi", "Framework web"),
        ("uvicorn", "Servidor ASGI")
    ]
    
    missing_deps = []
    
    for dep, description in critical_deps:
        try:
            if dep == "torch":
                import torch
                print(f"   ✅ {dep}: {torch.__version__} - {description}")
                print(f"      CUDA disponible: {torch.cuda.is_available()}")
            elif dep == "transformers":
                import transformers
                print(f"   ✅ {dep}: {transformers.__version__} - {description}")
            elif dep == "peft":
                import peft
                print(f"   ✅ {dep}: {peft.__version__} - {description}")
            elif dep == "datasets":
                import datasets
                print(f"   ✅ {dep}: {datasets.__version__} - {description}")
            elif dep == "fastapi":
                import fastapi
                print(f"   ✅ {dep}: {fastapi.__version__} - {description}")
            elif dep == "uvicorn":
                import uvicorn
                print(f"   ✅ {dep}: {uvicorn.__version__} - {description}")
        except ImportError:
            print(f"   ❌ {dep}: NO INSTALADO - {description}")
            missing_deps.append(dep)
    
    return missing_deps

def provide_solutions(missing_files, missing_deps, config_ok, basic_ai_ok, enhanced_ai_ok):
    """Proporciona soluciones basadas en los problemas encontrados"""
    print("\n🔧 SOLUCIONES RECOMENDADAS")
    print("=" * 50)
    
    if missing_deps:
        print("1️⃣ INSTALAR DEPENDENCIAS FALTANTES:")
        for dep in missing_deps:
            print(f"   pip install {dep}")
    
    if missing_files:
        print("\n2️⃣ CREAR ARCHIVOS FALTANTES:")
        for file in missing_files:
            if "enhanced_ai_service" in file:
                print(f"   ❌ {file} - Necesitas crear este archivo")
                print("   💡 Usa el código del artifact que te proporcioné")
            elif "model_config.json" in file:
                print(f"   ❌ {file} - Necesitas entrenar modelos")
                print("   💡 Ejecuta: python main_training.py")
    
    if not config_ok:
        print("\n3️⃣ ENTRENAR MODELOS:")
        print("   python main_training.py")
    
    if not basic_ai_ok:
        print("\n4️⃣ ARREGLAR SERVICIO BÁSICO:")
        print("   - Verificar app/services/ai_service.py")
        print("   - Verificar imports y dependencias")
    
    if not enhanced_ai_ok:
        print("\n5️⃣ ARREGLAR SERVICIO MEJORADO:")
        print("   - Verificar app/services/enhanced_ai_service.py")
        print("   - Verificar que herede correctamente de AIService")
    
    print("\n6️⃣ INICIAR APLICACIÓN:")
    print("   uvicorn app.main:app --reload")

async def main():
    """Función principal de diagnóstico"""
    print("🔍 DIAGNÓSTICO COMPLETO DEL SISTEMA")
    print("=" * 60)
    
    # Verificaciones
    missing_files = check_files_structure()
    missing_deps = check_dependencies()
    config_ok = check_model_config()
    
    basic_ai_ok = await test_basic_ai_service()
    enhanced_ai_ok = await test_enhanced_ai_service() if basic_ai_ok else False
    
    # Test API solo si los servicios funcionan
    if basic_ai_ok and enhanced_ai_ok:
        await test_api_endpoints()
    
    # Proporcionar soluciones
    provide_solutions(missing_files, missing_deps, config_ok, basic_ai_ok, enhanced_ai_ok)
    
    print("\n" + "=" * 60)
    if not missing_files and not missing_deps and config_ok and basic_ai_ok and enhanced_ai_ok:
        print("🎉 SISTEMA FUNCIONANDO CORRECTAMENTE")
    else:
        print("⚠️ SISTEMA NECESITA CORRECCIONES")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())