# cleanup_for_git.py - Script para limpiar archivos antes de Git
"""
Script para limpiar archivos innecesarios antes de hacer commit
"""

import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_folder_size(folder_path):
    """Calcula el tamaño de una carpeta en MB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
    except:
        return 0
    return total_size / (1024 * 1024)  # Convertir a MB

def cleanup_large_files():
    """Limpia archivos grandes que no deben estar en Git"""
    
    print("🧹 LIMPIANDO ARCHIVOS PARA GIT")
    print("=" * 50)
    
    # Carpetas a limpiar (GRANDES)
    large_folders = [
        "models/fine_tuned",
        "model_cache", 
        ".cache",
        "checkpoints",
        "runs",
        "outputs",
        "wandb",
        "data/datasets"
    ]
    
    # Archivos a limpiar
    files_to_clean = [
        "training.log",
        "*.log",
        "GUIA_MODELOS_FINE_TUNED.md",  # Se regenera automáticamente
        "validate_fine_tuned_models.py",  # Se regenera automáticamente
        "diagnose_model_modules.py",
        "install_fine_tuning.sh",
        "quick_start_fine_tuning.sh"
    ]
    
    total_freed = 0
    
    # Limpiar carpetas grandes
    for folder in large_folders:
        if os.path.exists(folder):
            size_mb = get_folder_size(folder)
            try:
                if folder == "models/fine_tuned":
                    # Mantener solo model_config.json
                    config_file = os.path.join(folder, "model_config.json")
                    if os.path.exists(config_file):
                        # Hacer backup del config
                        shutil.copy2(config_file, "model_config_backup.json")
                    
                    shutil.rmtree(folder)
                    os.makedirs(folder, exist_ok=True)
                    
                    # Restaurar config
                    if os.path.exists("model_config_backup.json"):
                        shutil.move("model_config_backup.json", config_file)
                        print(f"   ✅ {folder}/ limpiado (guardado model_config.json)")
                    else:
                        print(f"   ✅ {folder}/ limpiado completamente")
                else:
                    shutil.rmtree(folder)
                    print(f"   ✅ {folder}/ eliminado")
                
                total_freed += size_mb
                print(f"      Liberados: {size_mb:.1f} MB")
                
            except Exception as e:
                print(f"   ⚠️ No se pudo limpiar {folder}: {e}")
    
    # Limpiar archivos específicos
    for file_pattern in files_to_clean:
        if "*" in file_pattern:
            # Manejar patrones con wildcard
            import glob
            files = glob.glob(file_pattern)
            for file in files:
                try:
                    os.remove(file)
                    print(f"   ✅ {file} eliminado")
                except:
                    pass
        else:
            if os.path.exists(file_pattern):
                try:
                    os.remove(file_pattern)
                    print(f"   ✅ {file_pattern} eliminado")
                except Exception as e:
                    print(f"   ⚠️ No se pudo eliminar {file_pattern}: {e}")
    
    print(f"\n📊 TOTAL LIBERADO: {total_freed:.1f} MB")
    print("\n📋 ARCHIVOS IMPORTANTES MANTENIDOS:")
    print("   ✅ training/ (código de fine-tuning)")
    print("   ✅ app/services/enhanced_ai_service.py")
    print("   ✅ main_training.py")
    print("   ✅ integrate_fine_tuned_models.py")
    print("   ✅ requirements_fine_tuning.txt")
    print("   ✅ models/fine_tuned/model_config.json")

def show_git_status():
    """Muestra qué archivos están en Git"""
    print("\n📁 ARCHIVOS DE FINE-TUNING PARA GIT:")
    print("=" * 50)
    
    important_files = [
        "training/__init__.py",
        "training/lora_trainer.py", 
        "training/data_preparation.py",
        "main_training.py",
        "integrate_fine_tuned_models.py",
        "app/services/enhanced_ai_service.py",
        "requirements_fine_tuning.txt",
        "models/fine_tuned/model_config.json"
    ]
    
    for file in important_files:
        if os.path.exists(file):
            size_kb = os.path.getsize(file) / 1024
            print(f"   ✅ {file} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ {file} - FALTANTE")

def create_gitkeep_files():
    """Crea archivos .gitkeep para mantener estructura de carpetas"""
    folders_to_keep = [
        "models/fine_tuned",
        "uploads",
        "temp"
    ]
    
    for folder in folders_to_keep:
        os.makedirs(folder, exist_ok=True)
        gitkeep_path = os.path.join(folder, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, "w") as f:
                f.write("# Mantener esta carpeta en Git\n")
            print(f"   ✅ Creado {gitkeep_path}")

def main():
    """Función principal"""
    cleanup_large_files()
    create_gitkeep_files()
    show_git_status()
    
    print("\n🚀 COMANDOS RECOMENDADOS PARA GIT:")
    print("=" * 50)
    print("git add .")
    print("git commit -m 'feat: Implementar fine-tuning con LoRA para IA educativa'")
    print("git push")
    
    print("\n📝 NOTA: Los modelos fine-tuned NO se suben a Git")
    print("   Para compartir modelos, usar Hugging Face Hub o almacenamiento externo")

if __name__ == "__main__":
    main()