"""
Módulo de entrenamiento para fine-tuning con LoRA
"""

from .lora_trainer import LoRATrainer
from .data_preparation import EducationalDataGenerator

__all__ = ["LoRATrainer", "EducationalDataGenerator"]