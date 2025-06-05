# main_training.py - CORREGIDO PARA WINDOWS
import os
import sys
import time
import json
import logging
import torch
from pathlib import Path
from datetime import datetime

# CORREGIR encoding para Windows
import locale
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

# Configurar logging SIN EMOJIS para evitar errores de encoding
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Script principal para fine-tuning en 4 horas"""
    
    print("INICIANDO FINE-TUNING EDUCATIVO CON LORA")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # === FASE 1: PREPARACIÓN (30 min) ===
        logger.info("FASE 1: PREPARACIÓN DE DATOS")
        
        from training.data_preparation import EducationalDataGenerator
        
        data_generator = EducationalDataGenerator()
        
        # Generar datos sintéticos
        logger.info("Generando datos de entrenamiento...")
        summary_data = data_generator.generate_summary_data(n_samples=400)
        question_data = data_generator.generate_question_data(n_samples=400)
        feedback_data = data_generator.generate_feedback_data(n_samples=200)
        
        logger.info(f"Datos generados: {len(summary_data)} resúmenes, {len(question_data)} preguntas, {len(feedback_data)} feedback")
        
        # === FASE 2: FINE-TUNING RESÚMENES (1 hora) ===
        logger.info("FASE 2: FINE-TUNING MODELO DE RESÚMENES")
        
        from training.lora_trainer import LoRATrainer
        from peft import TaskType
        
        # Entrenar modelo de resúmenes
        summarizer_trainer = LoRATrainer(
            model_name="facebook/bart-base",
            task_type=TaskType.SEQ_2_SEQ_LM,
            model_type="summarizer"
        )
        
        train_data, val_data = summarizer_trainer.prepare_dataset(summary_data, validation_split=0.1)
        
        summarizer_metrics = summarizer_trainer.train(
            train_dataset=train_data,
            val_dataset=val_data,
            output_dir="./models/fine_tuned/summarizer_lora",
            num_epochs=2,
            batch_size=4,
            learning_rate=5e-4
        )
        
        logger.info(f"Resumen modelo entrenado. Loss: {summarizer_metrics['train_loss']:.4f}")
        
        # === FASE 3: FINE-TUNING PREGUNTAS (1 hora) ===
        logger.info("FASE 3: FINE-TUNING GENERADOR DE PREGUNTAS")
        
        question_trainer = LoRATrainer(
            model_name="google/flan-t5-base",
            task_type=TaskType.SEQ_2_SEQ_LM,
            model_type="question_gen"
        )
        
        train_data_qa, val_data_qa = question_trainer.prepare_dataset(question_data, validation_split=0.1)
        
        question_metrics = question_trainer.train(
            train_dataset=train_data_qa,
            val_dataset=val_data_qa,
            output_dir="./models/fine_tuned/question_gen_lora",
            num_epochs=2,
            batch_size=4,
            learning_rate=3e-4
        )
        
        logger.info(f"Generador de preguntas entrenado. Loss: {question_metrics['train_loss']:.4f}")
        
        # === FASE 4: FINE-TUNING FEEDBACK (1 hora) ===
        logger.info("FASE 4: FINE-TUNING GENERADOR DE FEEDBACK")
        
        feedback_trainer = LoRATrainer(
            model_name="google/flan-t5-base",
            task_type=TaskType.SEQ_2_SEQ_LM,
            model_type="feedback"
        )
        
        train_data_fb, val_data_fb = feedback_trainer.prepare_dataset(feedback_data, validation_split=0.1)
        
        feedback_metrics = feedback_trainer.train(
            train_dataset=train_data_fb,
            val_dataset=val_data_fb,
            output_dir="./models/fine_tuned/feedback_gen_lora",
            num_epochs=3,
            batch_size=6,
            learning_rate=4e-4
        )
        
        logger.info(f"Generador de feedback entrenado. Loss: {feedback_metrics['train_loss']:.4f}")
        
        # === FASE 5: GUARDADO E INTEGRACIÓN (30 min) ===
        logger.info("FASE 5: GUARDADO E INTEGRACIÓN")
        
        # Crear archivo de configuración del modelo
        model_config = {
            "models": {
                "summarizer": {
                    "base_model": "facebook/bart-base",
                    "lora_path": "./models/fine_tuned/summarizer_lora",
                    "metrics": summarizer_metrics
                },
                "question_generator": {
                    "base_model": "google/flan-t5-base",
                    "lora_path": "./models/fine_tuned/question_gen_lora",
                    "metrics": question_metrics
                },
                "feedback_generator": {
                    "base_model": "google/flan-t5-base",
                    "lora_path": "./models/fine_tuned/feedback_gen_lora",
                    "metrics": feedback_metrics
                }
            },
            "training_info": {
                "timestamp": datetime.now().isoformat(),
                "total_training_time": time.time() - start_time,
                "datasets": {
                    "summary_samples": len(summary_data),
                    "question_samples": len(question_data),
                    "feedback_samples": len(feedback_data)
                },
                "device_used": "cuda" if torch.cuda.is_available() else "cpu"
            }
        }
        
        # Guardar configuración
        os.makedirs("./models/fine_tuned", exist_ok=True)
        with open("./models/fine_tuned/model_config.json", "w", encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("FINE-TUNING COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"Tiempo total: {total_time/3600:.2f} horas")
        print(f"Modelos entrenados: 3 (Resúmenes, Preguntas, Feedback)")
        print(f"Ubicación: ./models/fine_tuned/")
        print(f"Configuración: ./models/fine_tuned/model_config.json")
        print("=" * 60)
        
        return model_config
        
    except Exception as e:
        logger.error(f"Error durante el fine-tuning: {e}")
        raise

if __name__ == "__main__":
    main()