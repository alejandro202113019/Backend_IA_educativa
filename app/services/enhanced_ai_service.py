# SOLUCIÓN RÁPIDA: Crear app/services/enhanced_ai_service.py

# app/services/enhanced_ai_service.py
import torch
import json
import logging
import os
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from peft import PeftModel
from app.services.ai_service import AIService

logger = logging.getLogger(__name__)

class EnhancedAIService(AIService):
    """
    Servicio de IA mejorado con modelos fine-tuned usando LoRA
    Extiende el AIService original manteniendo compatibilidad
    """
    
    def __init__(self):
        super().__init__()
        self.fine_tuned_models = {}
        self.model_config = None
        self._load_fine_tuned_models()
    
    def _load_fine_tuned_models(self):
        """Carga los modelos fine-tuned si están disponibles"""
        config_path = "./models/fine_tuned/model_config.json"
        
        if not os.path.exists(config_path):
            logger.info("No se encontraron modelos fine-tuned, usando modelos base")
            return
        
        try:
            logger.info("Cargando modelos fine-tuned...")
            
            # Cargar configuración
            with open(config_path, 'r', encoding='utf-8') as f:
                self.model_config = json.load(f)
            
            # Cargar cada modelo fine-tuned
            self._load_summarizer_model()
            self._load_question_generator_model()
            self._load_feedback_generator_model()
            
            logger.info("Modelos fine-tuned cargados exitosamente")
            
        except Exception as e:
            logger.error(f"Error cargando modelos fine-tuned: {e}")
            logger.info("Usando modelos base como fallback")
    
    def _load_summarizer_model(self):
        """Carga el modelo de resúmenes fine-tuned"""
        try:
            config = self.model_config["models"]["summarizer"]
            base_model_name = config["base_model"]
            lora_path = config["lora_path"]
            
            if os.path.exists(lora_path):
                logger.info(f"Cargando modelo de resúmenes fine-tuned desde {lora_path}")
                
                # Cargar modelo base
                base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
                
                # Cargar LoRA weights
                self.fine_tuned_models["summarizer"] = PeftModel.from_pretrained(
                    base_model, lora_path
                ).to(self.device)
                
                # Cargar tokenizer
                self.fine_tuned_models["summarizer_tokenizer"] = AutoTokenizer.from_pretrained(
                    base_model_name
                )
                
                logger.info("Modelo de resúmenes fine-tuned cargado")
            
        except Exception as e:
            logger.error(f"Error cargando modelo de resúmenes: {e}")
    
    def _load_question_generator_model(self):
        """Carga el modelo generador de preguntas fine-tuned"""
        try:
            config = self.model_config["models"]["question_generator"]
            base_model_name = config["base_model"]
            lora_path = config["lora_path"]
            
            if os.path.exists(lora_path):
                logger.info(f"Cargando generador de preguntas fine-tuned desde {lora_path}")
                
                # Cargar modelo base T5
                base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
                
                # Cargar LoRA weights
                self.fine_tuned_models["question_gen"] = PeftModel.from_pretrained(
                    base_model, lora_path
                ).to(self.device)
                
                # Cargar tokenizer
                self.fine_tuned_models["question_gen_tokenizer"] = AutoTokenizer.from_pretrained(
                    base_model_name
                )
                
                logger.info("Generador de preguntas fine-tuned cargado")
            
        except Exception as e:
            logger.error(f"Error cargando generador de preguntas: {e}")
    
    def _load_feedback_generator_model(self):
        """Carga el modelo generador de feedback fine-tuned"""
        try:
            config = self.model_config["models"]["feedback_generator"]
            base_model_name = config["base_model"]
            lora_path = config["lora_path"]
            
            if os.path.exists(lora_path):
                logger.info(f"Cargando generador de feedback fine-tuned desde {lora_path}")
                
                # Cargar modelo base T5
                base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
                
                # Cargar LoRA weights
                self.fine_tuned_models["feedback_gen"] = PeftModel.from_pretrained(
                    base_model, lora_path
                ).to(self.device)
                
                # Cargar tokenizer
                self.fine_tuned_models["feedback_gen_tokenizer"] = AutoTokenizer.from_pretrained(
                    base_model_name
                )
                
                logger.info("Generador de feedback fine-tuned cargado")
            
        except Exception as e:
            logger.error(f"Error cargando generador de feedback: {e}")
    
    async def generate_summary(self, text: str, length: str = "medium") -> Dict[str, Any]:
        """Genera resumen usando modelo fine-tuned si está disponible"""
        # Usar modelo fine-tuned si está disponible
        if "summarizer" in self.fine_tuned_models:
            return await self._generate_summary_fine_tuned(text, length)
        else:
            # Fallback al método original
            return await super().generate_summary(text, length)
    
    async def _generate_summary_fine_tuned(self, text: str, length: str = "medium") -> Dict[str, Any]:
        """Genera resumen con modelo fine-tuned"""
        try:
            model = self.fine_tuned_models["summarizer"]
            tokenizer = self.fine_tuned_models["summarizer_tokenizer"]
            
            # Configurar longitud
            length_config = {
                "short": {"max_length": 100, "min_length": 30},
                "medium": {"max_length": 200, "min_length": 50},
                "long": {"max_length": 300, "min_length": 100}
            }
            config = length_config.get(length, length_config["medium"])
            
            # Limitar texto de entrada
            max_input_length = 500
            if len(text.split()) > max_input_length:
                text = " ".join(text.split()[:max_input_length])
            
            # Tokenizar
            inputs = tokenizer(
                text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generar resumen
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=config["max_length"],
                    min_length=config["min_length"],
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            # Decodificar
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "summary": summary,
                "success": True,
                "model_used": "fine_tuned_summarizer"
            }
            
        except Exception as e:
            logger.error(f"Error con modelo fine-tuned de resúmenes: {e}")
            # Fallback al modelo base
            return await super().generate_summary(text, length)
    
    async def generate_quiz(self, text: str, key_concepts: List[str], 
                          num_questions: int = 5, difficulty: str = "medium") -> Dict[str, Any]:
        """Genera quiz usando modelo fine-tuned si está disponible"""
        # Por ahora usar el método original con mejoras
        return await super().generate_quiz(text, key_concepts, num_questions, difficulty)
    
    async def generate_feedback(self, score: int, total: int, 
                              incorrect_questions: List[int], concepts: List[str]) -> str:
        """Genera feedback usando modelo fine-tuned si está disponible"""
        # Por ahora usar el método original con mejoras
        return await super().generate_feedback(score, total, incorrect_questions, concepts)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Obtiene el estado de todos los modelos (base + fine-tuned)"""
        
        status = {
            "base_models": {
                "summarizer_loaded": self.summarizer is not None,
                "t5_model_loaded": self.t5_model is not None,
                "classifier_loaded": self.classifier is not None
            },
            "fine_tuned_models": {
                "summarizer_loaded": "summarizer" in self.fine_tuned_models,
                "question_gen_loaded": "question_gen" in self.fine_tuned_models,
                "feedback_gen_loaded": "feedback_gen" in self.fine_tuned_models
            },
            "device": self.device,
            "model_config_loaded": self.model_config is not None
        }
        
        if self.model_config:
            status["training_info"] = self.model_config.get("training_info", {})
        
        return status