#!/usr/bin/env python3
"""
diagnose_project.py - Diagnóstico completo del proyecto
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_project_structure():
    """Verificar estructura del proyecto"""
    print("🏗️  VERIFICANDO ESTRUCTURA DEL PROYECTO")
    print("=" * 50)
    
    required_dirs = [
        "app",
        "app/api",
        "app/api/routes", 
        "app/core",
        "app/models",
        "app/services",
        "app/utils",
        "models",
        "models/fine_tuned",
        "training",
        "uploads",
        "temp"
    ]
    
    required_files = [
        "app/main.py",
        "app/services/ai_service.py",
        "app/services/enhanced_ai_service.py",
        "app/services/service_manager.py",
        "app/services/nlp_service.py",
        "requirements.txt",
        "main_training.py"
    ]
    
    issues = []
    
    # Verificar directorios
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ - FALTANTE")
            issues.append(f"Directorio faltante: {dir_path}")
    
    # Verificar archivos
    for file_path in required_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size / 1024
            print(f"   ✅ {file_path} ({size:.1f} KB)")
        else:
            print(f"   ❌ {file_path} - FALTANTE")
            issues.append(f"Archivo faltante: {file_path}")
    
    # Verificar duplicados problemáticos
    if Path("model").exists() and Path("models").exists():
        print(f"   ⚠️  PROBLEMA: Ambas carpetas 'model' y 'models' existen")
        issues.append("Conflicto: carpetas 'model' y 'models' duplicadas")
    
    return issues

def check_python_dependencies():
    """Verificar dependencias de Python"""
    print("\n📦 VERIFICANDO DEPENDENCIAS DE PYTHON")
    print("=" * 50)
    
    critical_packages = [
        "fastapi",
        "uvicorn", 
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
        "spacy"
    ]
    
    missing_packages = []
    
    for package in critical_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NO INSTALADO")
            missing_packages.append(package)
    
    return missing_packages

def check_model_config():
    """Verificar configuración de modelos"""
    print("\n🤖 VERIFICANDO CONFIGURACIÓN DE MODELOS")
    print("=" * 50)
    
    issues = []
    
    # Verificar model_config.json
    config_path = Path("models/fine_tuned/model_config.json")
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"   ✅ model_config.json cargado")
            
            # Verificar estructura
            if "models" in config:
                for model_name, model_info in config["models"].items():
                    lora_path = model_info.get("lora_path", "")
                    if Path(lora_path).exists():
                        print(f"   ✅ {model_name}: Modelo fine-tuned encontrado")
                    else:
                        print(f"   ⚠️  {model_name}: Solo modelo base (fine-tuned no entrenado)")
            
        except Exception as e:
            print(f"   ❌ Error leyendo model_config.json: {e}")
            issues.append("model_config.json corrupto")
    else:
        print(f"   ⚠️  model_config.json no encontrado")
        issues.append("model_config.json faltante")
    
    # Verificar caché de modelos
    cache_dir = Path("model_cache")
    if cache_dir.exists():
        cache_files = list(cache_dir.rglob("*"))
        cache_size = sum(f.stat().st_size for f in cache_files if f.is_file()) / (1024**2)
        print(f"   ✅ Caché de modelos: {len(cache_files)} archivos, {cache_size:.1f} MB")
    else:
        print(f"   ⚠️  Caché de modelos vacío")
    
    return issues

def test_service_loading():
    """Probar carga de servicios"""
    print("\n🔧 PROBANDO CARGA DE SERVICIOS")
    print("=" * 50)
    
    issues = []
    
    try:
        # Test 1: ServiceManager
        print("   🔄 Probando ServiceManager...")
        sys.path.insert(0, str(Path.cwd()))
        from app.services.service_manager import service_manager
        print("   ✅ ServiceManager importado")
        
        # Test 2: AI Service
        print("   🔄 Probando AIService...")
        try:
            ai_service = service_manager.ai_service
            print("   ✅ AIService cargado")
        except Exception as e:
            print(f"   ❌ Error cargando AIService: {e}")
            issues.append(f"Error AIService: {e}")
        
        # Test 3: NLP Service  
        print("   🔄 Probando NLPService...")
        try:
            nlp_service = service_manager.nlp_service
            print("   ✅ NLPService cargado")
        except Exception as e:
            print(f"   ❌ Error cargando NLPService: {e}")
            issues.append(f"Error NLPService: {e}")
            
    except Exception as e:
        print(f"   ❌ Error importando servicios: {e}")
        issues.append(f"Error importación: {e}")
    
    return issues

def check_environment():
    """Verificar configuración del entorno"""
    print("\n🌍 VERIFICANDO ENTORNO")
    print("=" * 50)
    
    # Python version
    python_version = sys.version.split()[0]
    print(f"   🐍 Python: {python_version}")
    
    # CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   🎮 GPU: {gpu_name}")
        else:
            print(f"   💻 GPU: No disponible (CPU)")
    except:
        print(f"   ❓ GPU: No se pudo verificar")
    
    # Variables de entorno
    env_file = Path(".env")
    if env_file.exists():
        print(f"   ✅ Archivo .env encontrado")
    else:
        print(f"   ⚠️  Archivo .env no encontrado")
    
    return []

def generate_fix_recommendations(all_issues):
    """Generar recomendaciones de reparación"""
    print("\n🔧 RECOMENDACIONES DE REPARACIÓN")
    print("=" * 50)
    
    if not all_issues:
        print("   🎉 ¡No se encontraron problemas críticos!")
        print("   ✅ El proyecto parece estar bien configurado")
        return
    
    print("   📋 Problemas encontrados:")
    for i, issue in enumerate(all_issues, 1):
        print(f"   {i}. {issue}")
    
    print("\n   🚀 PASOS PARA REPARAR:")
    
    # Paso 1: Estructura
    if any("faltante" in issue for issue in all_issues):
        print("   1. Ejecutar script de reparación de estructura:")
        print("      bash fix_project.sh")
    
    # Paso 2: Dependencias
    if any("NO INSTALADO" in issue for issue in all_issues):
        print("   2. Instalar dependencias faltantes:")
        print("      pip install -r requirements.txt")
        print("      python -m spacy download es_core_news_sm")
    
    # Paso 3: Configuración
    if any("config" in issue.lower() for issue in all_issues):
        print("   3. Reparar configuración:")
        print("      python integrate_fine_tuned_models.py")
    
    # Paso 4: Modelos
    if any("modelo" in issue.lower() for issue in all_issues):
        print("   4. Entrenar modelos fine-tuned (opcional):")
        print("      python main_training.py")
    
    print("\n   ✅ Después de las reparaciones:")
    print("      python test_models.py")
    print("      uvicorn app.main:app --reload")

def main():
    """Función principal de diagnóstico"""
    print("🔍 DIAGNÓSTICO COMPLETO DEL PROYECTO DE IA EDUCATIVA")
    print("=" * 60)
    print(f"📁 Directorio: {Path.cwd()}")
    print()
    
    all_issues = []
    
    # Ejecutar todas las verificaciones
    all_issues.extend(check_project_structure())
    missing_packages = check_python_dependencies()
    all_issues.extend([f"Paquete no instalado: {pkg}" for pkg in missing_packages])
    all_issues.extend(check_model_config())
    all_issues.extend(test_service_loading())
    check_environment()
    
    # Generar recomendaciones
    generate_fix_recommendations(all_issues)
    
    print(f"\n📊 RESUMEN: {len(all_issues)} problemas encontrados")
    
    if len(all_issues) == 0:
        print("🎉 ¡PROYECTO EN EXCELENTE ESTADO!")
        print("🚀 Puedes ejecutar directamente: uvicorn app.main:app --reload")
        return 0
    elif len(all_issues) <= 3:
        print("⚠️  PROBLEMAS MENORES - Fáciles de solucionar")
        return 1
    else:
        print("❌ PROBLEMAS IMPORTANTES - Requiere atención")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)