# training/lora_trainer.py - CORREGIDO FINAL
import torch
import json
import logging
import os
from typing import Dict, List, Any, Optional
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    T5ForConditionalGeneration, T5Tokenizer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from datetime import datetime

logger = logging.getLogger(__name__)

class LoRATrainer:
    """
    Trainer especializado para fine-tuning educativo con LoRA
    """
    def __init__(self, model_name: str, task_type: TaskType, model_type: str = "general"):
        self.model_name = model_name
        self.task_type = task_type
        self.model_type = model_type  # "summarizer", "question_gen", "feedback"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Inicializando LoRATrainer para {model_type}")
        logger.info(f"Dispositivo: {self.device}")
        
        # Cargar modelo base y tokenizer
        self._load_base_model()
        
        # Configurar LoRA
        self._setup_lora()
    
    def _load_base_model(self):
        """Carga el modelo base y tokenizer"""
        try:
            logger.info(f"Cargando modelo base: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if "t5" in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Configurar tokens especiales si es necesario
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Modelo base cargado exitosamente")
            
        except Exception as e:
            logger.error(f"Error cargando modelo base: {e}")
            raise
    
    def _get_target_modules(self, model_name: str) -> List[str]:
        """Obtiene los módulos objetivo correctos según el modelo"""
        
        if "bart" in model_name.lower():
            # BART usa estos nombres de módulos
            return ["q_proj", "v_proj", "k_proj", "out_proj"]
        elif "t5" in model_name.lower():
            # T5 usa estos nombres
            return ["q", "v", "k", "o", "wi", "wo"]
        else:
            # Fallback genérico
            return ["q_proj", "v_proj"]
    
    def _setup_lora(self):
        """Configura LoRA para el modelo"""
        try:
            # Obtener módulos objetivo correctos
            target_modules = self._get_target_modules(self.model_name)
            
            # Configuración LoRA optimizada por tipo de modelo
            lora_configs = {
                "summarizer": LoraConfig(
                    task_type=self.task_type,
                    r=16,  # Rank alto para resúmenes complejos
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=target_modules
                ),
                "question_gen": LoraConfig(
                    task_type=self.task_type,
                    r=12,  # Rank medio para generación de preguntas
                    lora_alpha=24,
                    lora_dropout=0.05,
                    target_modules=target_modules
                ),
                "feedback": LoraConfig(
                    task_type=self.task_type,
                    r=8,   # Rank menor para feedback
                    lora_alpha=16,
                    lora_dropout=0.1,
                    target_modules=target_modules
                )
            }
            
            config = lora_configs.get(self.model_type, lora_configs["summarizer"])
            
            logger.info(f"Configurando LoRA con r={config.r}, alpha={config.lora_alpha}")
            logger.info(f"Módulos objetivo: {target_modules}")
            
            self.model = get_peft_model(self.model, config)
            self.model.print_trainable_parameters()
            
            logger.info("LoRA configurado exitosamente")
            
        except Exception as e:
            logger.error(f"Error configurando LoRA: {e}")
            raise
    
    def prepare_dataset(self, data_list: List[Dict], validation_split: float = 0.1) -> tuple:
        """Prepara dataset para entrenamiento"""
        try:
            logger.info(f"Preparando dataset con {len(data_list)} ejemplos")
            
            def tokenize_function(examples):
                # Configuración específica por tipo de modelo
                max_input_length = 512
                max_target_length = 150
                
                if self.model_type == "summarizer":
                    max_target_length = 200
                elif self.model_type == "question_gen":
                    max_target_length = 100
                elif self.model_type == "feedback":
                    max_target_length = 300
                
                # CORREGIR: Asegurar que son listas de strings
                input_texts = examples["input_text"]
                target_texts = examples["target_text"]
                
                # Verificar que son listas
                if not isinstance(input_texts, list):
                    input_texts = [input_texts]
                if not isinstance(target_texts, list):
                    target_texts = [target_texts]
                
                # Tokenizar inputs - SIN return_tensors aquí
                model_inputs = self.tokenizer(
                    input_texts,
                    max_length=max_input_length,
                    truncation=True,
                    padding=True
                )
                
                # Tokenizar targets - SIN return_tensors aquí
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        target_texts,
                        max_length=max_target_length,
                        truncation=True,
                        padding=True
                    )
                
                # Asignar labels
                model_inputs["labels"] = labels["input_ids"]
                
                return model_inputs
            
            # Crear dataset
            dataset = Dataset.from_list(data_list)
            
            # Dividir en entrenamiento y validación
            if validation_split > 0:
                split_dataset = dataset.train_test_split(test_size=validation_split, seed=42)
                train_dataset = split_dataset["train"]
                val_dataset = split_dataset["test"]
            else:
                train_dataset = dataset
                val_dataset = None
            
            # Tokenizar - CRUCIAL: remove_columns para eliminar columnas originales
            columns_to_remove = train_dataset.column_names
            train_dataset = train_dataset.map(
                tokenize_function, 
                batched=True,
                remove_columns=columns_to_remove
            )
            
            if val_dataset:
                val_dataset = val_dataset.map(
                    tokenize_function, 
                    batched=True,
                    remove_columns=columns_to_remove
                )
            
            logger.info(f"Dataset preparado: {len(train_dataset)} entrenamiento, {len(val_dataset) if val_dataset else 0} validación")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Error preparando dataset: {e}")
            raise
    
    # training/lora_trainer.py - SOLO CORREGIR LA FUNCIÓN train() al final

    def train(self, train_dataset, val_dataset=None, output_dir="./models/fine_tuned", 
              num_epochs=3, batch_size=4, learning_rate=5e-4) -> Dict[str, Any]:
        """Entrena el modelo con LoRA"""
        try:
            logger.info(f"Iniciando entrenamiento...")
            
            # Configurar argumentos de entrenamiento
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=4,
                learning_rate=learning_rate,
                warmup_steps=100,
                weight_decay=0.01,
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="epoch" if val_dataset else "no",
                save_total_limit=2,
                load_best_model_at_end=True if val_dataset else False,
                metric_for_best_model="eval_loss" if val_dataset else None,
                greater_is_better=False,
                report_to="none",
                fp16=torch.cuda.is_available(),
                dataloader_num_workers=0,  # Para Windows
                remove_unused_columns=False,
            )
            
            # Data collator
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                return_tensors="pt",
                padding=True
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Entrenar
            logger.info("Comenzando entrenamiento...")
            train_result = trainer.train()
            
            # Guardar modelo
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # CORREGIR: Usar los atributos correctos del TrainOutput
            metrics = {
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics.get("train_runtime", 0),  # CORREGIDO
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
                "model_type": self.model_type,
                "base_model": self.model_name,
                "training_samples": len(train_dataset),
                "epochs": num_epochs,
                "timestamp": datetime.now().isoformat()
            }
            
            if val_dataset:
                eval_result = trainer.evaluate()
                metrics["eval_loss"] = eval_result["eval_loss"]
                metrics["eval_runtime"] = eval_result.get("eval_runtime", 0)
            
            # Guardar métricas
            with open(os.path.join(output_dir, "training_metrics.json"), "w", encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Entrenamiento completado. Loss final: {train_result.training_loss:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error durante entrenamiento: {e}")
            raise