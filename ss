# app/services/enhanced_ai_service.py - VERSIÓN FINAL MEJORADA PARA CUALQUIER TEXTO
import torch
import json
import logging
import os
import re
import random
from typing import Dict, Any, List, Optional
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration,
    pipeline, BartForConditionalGeneration, BartTokenizer
)
from peft import PeftModel
from app.services.ai_service import AIService
from collections import Counter

logger = logging.getLogger(__name__)

class EnhancedAIService(AIService):
    """
    Servicio de IA mejorado con modelos fine-tuned y lógica perfeccionada para cualquier tipo de texto
    """
    
    def __init__(self):
        # Llamar al constructor padre
        super().__init__()
        self.fine_tuned_models = {}
        self.model_config = None
        
        # Cargar modelos mejorados
        self._load_enhanced_models()
        
        # Patrones y plantillas universales
        self._setup_universal_patterns()
    
    def _setup_universal_patterns(self):
        """Configura patrones universales para diferentes tipos de texto"""
        self.text_patterns = {
            "historia": {
                "keywords": ["guerra", "batalla", "imperio", "revolución", "siglo", "año", "época", "dinastía"],
                "question_types": ["cronológico", "causal", "comparativo", "factual"],
                "summary_focus": ["fechas", "personajes", "eventos", "causas", "consecuencias"]
            },
            "ciencia": {
                "keywords": ["proceso", "método", "experimento", "hipótesis", "teoría", "ley", "fórmula", "análisis"],
                "question_types": ["conceptual", "procedimental", "aplicativo", "analítico"],
                "summary_focus": ["conceptos", "procesos", "aplicaciones", "características"]
            },
            "tecnología": {
                "keywords": ["algoritmo", "sistema", "datos", "software", "hardware", "red", "código", "digital"],
                "question_types": ["técnico", "aplicativo", "comparativo", "funcional"],
                "summary_focus": ["funcionalidades", "aplicaciones", "ventajas", "componentes"]
            },
            "literatura": {
                "keywords": ["personaje", "narrador", "trama", "estilo", "género", "obra", "autor", "movimiento"],
                "question_types": ["interpretativo", "analítico", "contextual", "estilístico"],
                "summary_focus": ["temas", "personajes", "estilo", "contexto"]
            },
            "economia": {
                "keywords": ["mercado", "precio", "oferta", "demanda", "inversión", "capital", "producto", "empresa"],
                "question_types": ["analítico", "aplicativo", "comparativo", "predictivo"],
                "summary_focus": ["conceptos", "relaciones", "factores", "impacto"]
            },
            "medicina": {
                "keywords": ["síntoma", "diagnóstico", "tratamiento", "enfermedad", "paciente", "terapia", "prevención"],
                "question_types": ["diagnóstico", "terapéutico", "preventivo", "factual"],
                "summary_focus": ["síntomas", "causas", "tratamientos", "prevención"]
            },
            "filosofía": {
                "keywords": ["concepto", "idea", "pensamiento", "razón", "ética", "moral", "existencia", "conocimiento"],
                "question_types": ["conceptual", "crítico", "comparativo", "reflexivo"],
                "summary_focus": ["conceptos", "argumentos", "posturas", "implicaciones"]
            }
        }
    
    def _load_enhanced_models(self):
        """Carga modelos mejorados (fine-tuned si existen, base optimizado si no)"""
        config_path = "./models/fine_tuned/model_config.json"
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.model_config = json.load(f)
                self._load_fine_tuned_models()
                logger.info("Modelos fine-tuned cargados exitosamente")
            except Exception as e:
                logger.warning(f"Error cargando modelos fine-tuned: {e}")
                self._setup_enhanced_base_models()
        else:
            logger.info("No se encontraron modelos fine-tuned, usando modelos base optimizados")
            self._setup_enhanced_base_models()
    
    def _setup_enhanced_base_models(self):
        """Configura modelos base optimizados"""
        try:
            # Usar modelos base pero con configuraciones optimizadas
            logger.info("Configurando modelos base optimizados...")
            
            # Para resúmenes: BART optimizado
            self.enhanced_summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1,
                max_length=300,
                min_length=100,
                do_sample=True,
                temperature=0.7
            )
            
            # Para generación de texto: T5 optimizado
            self.enhanced_generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=0 if self.device == "cuda" else -1,
                max_length=150,
                do_sample=True,
                temperature=0.8
            )
            
            logger.info("Modelos base optimizados configurados")
            
        except Exception as e:
            logger.error(f"Error configurando modelos base: {e}")
    
    def _load_fine_tuned_models(self):
        """Carga los modelos fine-tuned si están disponibles"""
        try:
            models = self.model_config.get("models", {})
            
            for model_name, config in models.items():
                lora_path = config.get("lora_path")
                if os.path.exists(lora_path):
                    try:
                        base_model_name = config["base_model"]
                        
                        if "bart" in base_model_name.lower():
                            base_model = BartForConditionalGeneration.from_pretrained(base_model_name)
                            tokenizer = BartTokenizer.from_pretrained(base_model_name)
                        else:
                            base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
                            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                        
                        fine_tuned_model = PeftModel.from_pretrained(base_model, lora_path).to(self.device)
                        
                        self.fine_tuned_models[model_name] = {
                            "model": fine_tuned_model,
                            "tokenizer": tokenizer
                        }
                        
                        logger.info(f"Modelo fine-tuned cargado: {model_name}")
                        
                    except Exception as e:
                        logger.warning(f"Error cargando modelo {model_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error general cargando modelos fine-tuned: {e}")
    
    def _detect_text_type(self, text: str) -> str:
        """Detecta el tipo de texto basado en contenido y keywords"""
        text_lower = text.lower()
        
        # Contar keywords por categoría
        category_scores = {}
        
        for category, patterns in self.text_patterns.items():
            score = 0
            for keyword in patterns["keywords"]:
                score += text_lower.count(keyword)
            category_scores[category] = score
        
        # Detectores específicos adicionales
        if any(word in text_lower for word in ["guerra", "batalla", "siglo", "año", "emperador", "rey"]):
            category_scores["historia"] += 5
        
        if any(word in text_lower for word in ["célula", "gen", "proteína", "bacteria", "virus"]):
            category_scores["ciencia"] += 5
        
        if any(word in text_lower for word in ["algoritmo", "software", "programa", "código", "datos"]):
            category_scores["tecnología"] += 5
        
        if any(word in text_lower for word in ["mercado", "económico", "precio", "empresa", "negocio"]):
            category_scores["economia"] += 5
        
        # Retornar categoría con mayor score o "general"
        best_category = max(category_scores, key=category_scores.get)
        return best_category if category_scores[best_category] > 0 else "general"
    
    async def generate_summary(self, text: str, length: str = "medium") -> Dict[str, Any]:
        """Genera resumen inteligente adaptado al tipo de texto"""
        try:
            # Detectar tipo de texto
            text_type = self._detect_text_type(text)
            logger.info(f"Tipo de texto detectado: {text_type}")
            
            # Usar modelo fine-tuned si está disponible
            if "summarizer" in self.fine_tuned_models:
                return await self._generate_fine_tuned_summary(text, length, text_type)
            else:
                return await self._generate_enhanced_summary(text, length, text_type)
                
        except Exception as e:
            logger.error(f"Error generando resumen: {e}")
            return await super().generate_summary(text, length)
    
    async def _generate_fine_tuned_summary(self, text: str, length: str, text_type: str) -> Dict[str, Any]:
        """Genera resumen con modelo fine-tuned"""
        try:
            model_info = self.fine_tuned_models["summarizer"]
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Crear prompt especializado según tipo de texto
            prompt = self._create_specialized_prompt(text, text_type, "summary")
            
            # Configurar longitud
            length_config = {
                "short": {"max_length": 150, "min_length": 50},
                "medium": {"max_length": 250, "min_length": 100},
                "long": {"max_length": 350, "min_length": 150}
            }
            config = length_config.get(length, length_config["medium"])
            
            # Tokenizar
            inputs = tokenizer(
                prompt,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generar
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=config["max_length"],
                    min_length=config["min_length"],
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            # Decodificar y procesar
            raw_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_summary = self._create_structured_summary(raw_summary, text, text_type)
            
            return {
                "summary": final_summary,
                "success": True,
                "model_used": "fine_tuned_summarizer",
                "text_type": text_type
            }
            
        except Exception as e:
            logger.error(f"Error con modelo fine-tuned: {e}")
            return await self._generate_enhanced_summary(text, length, text_type)
    
    async def _generate_enhanced_summary(self, text: str, length: str, text_type: str) -> Dict[str, Any]:
        """Genera resumen mejorado con modelos base"""
        try:
            # Usar el modelo base optimizado
            if hasattr(self, 'enhanced_summarizer'):
                # Generar resumen base
                summary_result = self.enhanced_summarizer(
                    text[:1024],  # Limitar entrada
                    max_length=200 if length == "medium" else (150 if length == "short" else 250),
                    min_length=80 if length == "medium" else (50 if length == "short" else 120),
                    do_sample=True,
                    temperature=0.7
                )
                raw_summary = summary_result[0]['summary_text']
            else:
                # Fallback al método padre
                result = await super().generate_summary(text, length)
                raw_summary = result["summary"]
            
            # Crear resumen estructurado
            final_summary = self._create_structured_summary(raw_summary, text, text_type)
            
            return {
                "summary": final_summary,
                "success": True,
                "model_used": "enhanced_base_summarizer",
                "text_type": text_type
            }
            
        except Exception as e:
            logger.error(f"Error en resumen mejorado: {e}")
            return await super().generate_summary(text, length)
    
    def _create_specialized_prompt(self, text: str, text_type: str, task: str) -> str:
        """Crea prompts especializados según tipo de texto y tarea"""
        
        if task == "summary":
            if text_type == "historia":
                return f"""Crea un resumen educativo de este texto histórico, destacando:
- Fechas y períodos importantes
- Personajes clave y sus roles
- Eventos principales y su secuencia
- Causas y consecuencias

TEXTO: {text[:800]}

RESUMEN EDUCATIVO:"""

            elif text_type == "ciencia":
                return f"""Crea un resumen educativo de este texto científico, destacando:
- Conceptos y principios fundamentales
- Procesos o métodos descritos
- Aplicaciones prácticas
- Características importantes

TEXTO: {text[:800]}

RESUMEN EDUCATIVO:"""

            elif text_type == "tecnología":
                return f"""Crea un resumen educativo de este texto tecnológico, destacando:
- Tecnologías y sistemas principales
- Funcionalidades y características
- Aplicaciones y usos
- Ventajas y beneficios

TEXTO: {text[:800]}

RESUMEN EDUCATIVO:"""

            else:
                return f"""Crea un resumen educativo estructurado de este texto:

TEXTO: {text[:800]}

RESUMEN EDUCATIVO:"""
        
        elif task == "question":
            return f"""Genera preguntas educativas de calidad sobre este texto de {text_type}:

TEXTO: {text[:600]}

Pregunta educativa:"""
        
        return text
    
    def _create_structured_summary(self, raw_summary: str, original_text: str, text_type: str) -> str:
        """Crea resumen estructurado según el tipo de texto"""
        
        # Extraer información específica del texto
        key_info = self._extract_specialized_info(original_text, text_type)
        
        # Crear estructura base
        structured_summary = "📚 **RESUMEN EDUCATIVO ESPECIALIZADO**\n\n"
        
        # Agregar tema principal
        structured_summary += f"🎯 **TEMA:** {key_info.get('topic', 'Análisis del contenido')}\n\n"
        
        # Agregar información específica según tipo
        if text_type == "historia" and key_info.get('dates'):
            structured_summary += f"📅 **CRONOLOGÍA:** {' → '.join(key_info['dates'][:4])}\n\n"
        
        if key_info.get('key_concepts'):
            structured_summary += f"🔑 **CONCEPTOS CLAVE:** {', '.join(key_info['key_concepts'][:5])}\n\n"
        
        if text_type == "historia" and key_info.get('people'):
            structured_summary += f"👥 **FIGURAS IMPORTANTES:** {', '.join(key_info['people'][:4])}\n\n"
        
        # Limpiar y agregar contenido principal
        clean_summary = self._clean_and_improve_summary(raw_summary)
        structured_summary += f"📝 **CONTENIDO PRINCIPAL:**\n{clean_summary}\n\n"
        
        # Agregar puntos clave específicos
        key_points = self._generate_key_points(original_text, text_type)
        if key_points:
            structured_summary += f"💡 **PUNTOS CLAVE:**\n"
            for i, point in enumerate(key_points, 1):
                structured_summary += f"{i}. {point}\n"
        
        return structured_summary
    
    def _extract_specialized_info(self, text: str, text_type: str) -> Dict[str, Any]:
        """Extrae información especializada según el tipo de texto"""
        info = {
            "topic": "el tema principal",
            "key_concepts": [],
            "dates": [],
            "people": [],
            "processes": [],
            "locations": []
        }
        
        if text_type == "historia":
            # Extraer fechas y años
            dates = re.findall(r'\b((?:siglo\s+)?(?:XV{0,3}I{0,3}|I{1,3}V?|V|IX|IV|X{1,3})|(?:19|20)\d{2}|\d{1,2}\s+de\s+\w+\s+de\s+\d{4})\b', text, re.IGNORECASE)
            info["dates"] = list(set(dates))[:5]
            
            # Extraer nombres propios (posibles personajes históricos)
            names = re.findall(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+){1,2}\b', text)
            info["people"] = list(set([name for name in names if len(name.split()) <= 3]))[:5]
            
            # Detectar tema histórico
            if "guerra mundial" in text.lower():
                info["topic"] = "Guerra Mundial"
            elif "revolución" in text.lower():
                info["topic"] = "Revolución Histórica"
            elif "imperio" in text.lower():
                info["topic"] = "Historia Imperial"
        
        elif text_type == "ciencia":
            # Extraer procesos científicos
            processes = re.findall(r'\b\w*(?:ción|sis|oma|ema)\b', text, re.IGNORECASE)
            info["processes"] = list(set([p for p in processes if len(p) > 4]))[:5]
            
            # Detectar tema científico
            if "fotosíntesis" in text.lower():
                info["topic"] = "Fotosíntesis"
            elif "célula" in text.lower():
                info["topic"] = "Biología Celular"
            elif "química" in text.lower():
                info["topic"] = "Procesos Químicos"
        
        elif text_type == "tecnología":
            # Extraer términos técnicos
            tech_terms = re.findall(r'\b(?:algoritmo|software|hardware|sistema|programa|aplicación|tecnología)\w*\b', text, re.IGNORECASE)
            info["processes"] = list(set(tech_terms))[:5]
            
            if "inteligencia artificial" in text.lower():
                info["topic"] = "Inteligencia Artificial"
            elif "programación" in text.lower():
                info["topic"] = "Programación y Desarrollo"
        
        # Extraer conceptos clave generales
        words = re.findall(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{3,}\b', text)
        word_freq = Counter(words)
        stop_words = {'Para', 'Esto', 'Todo', 'Cada', 'Debe', 'Puede', 'Será', 'Está', 'Hace', 'Tiene'}
        concepts = [word for word, freq in word_freq.most_common(10) 
                   if word not in stop_words and freq > 1]
        info["key_concepts"] = concepts[:6]
        
        return info
    
    def _clean_and_improve_summary(self, summary: str) -> str:
        """Limpia y mejora el resumen generado"""
        # Eliminar errores comunes
        corrections = {
            "seguirra": "guerra",
            "eusu": "EEUU",
            "histororia": "historia",
            "teh": "the",
            "proces": "proceso"
        }
        
        clean_summary = summary
        for error, correction in corrections.items():
            clean_summary = clean_summary.replace(error, correction)
        
        # Mejorar estructura
        sentences = clean_summary.split('.')
        improved_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                # Capitalizar primera letra
                sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
                improved_sentences.append(sentence)
        
        return '. '.join(improved_sentences) + '.'
    
    def _generate_key_points(self, text: str, text_type: str) -> List[str]:
        """Genera puntos clave específicos según el tipo de texto"""
        points = []
        
        if text_type == "historia":
            if "guerra mundial" in text.lower():
                points = [
                    "Conflicto global que transformó el orden mundial",
                    "Involucró a múltiples naciones y continentes",
                    "Tuvo consecuencias políticas, sociales y económicas duraderas"
                ]
            elif "revolución" in text.lower():
                points = [
                    "Período de cambios sociales y políticos significativos",
                    "Transformó las estructuras de poder existentes",
                    "Influyó en el desarrollo histórico posterior"
                ]
        
        elif text_type == "ciencia":
            if "fotosíntesis" in text.lower():
                points = [
                    "Proceso fundamental para la vida en la Tierra",
                    "Convierte la energía solar en energía química",
                    "Produce oxígeno esencial para la respiración"
                ]
            else:
                points = [
                    "Conceptos científicos basados en evidencia y método",
                    "Aplicaciones prácticas en la vida cotidiana",
                    "Importancia para el avance del conocimiento"
                ]
        
        elif text_type == "tecnología":
            points = [
                "Tecnología que mejora la eficiencia y capacidades humanas",
                "Aplicaciones en múltiples sectores y disciplinas",
                "Evolución constante y adaptación a nuevas necesidades"
            ]
        
        # Puntos generales si no hay específicos
        if not points:
            sentences = text.split('.')
            important_sentences = [s.strip() for s in sentences 
                                 if 40 < len(s.strip()) < 120 and 
                                 any(word in s.lower() for word in ['importante', 'principal', 'fundamental', 'clave'])]
            points = important_sentences[:3]
        
        return points[:3]
    
    async def generate_quiz(self, text: str, key_concepts: List[str], 
                          num_questions: int = 5, difficulty: str = "medium") -> Dict[str, Any]:
        """Genera quiz inteligente adaptado al tipo de texto"""
        try:
            # Detectar tipo de texto
            text_type = self._detect_text_type(text)
            logger.info(f"Generando quiz para texto tipo: {text_type}")
            
            # Usar modelo fine-tuned si está disponible
            if "question_generator" in self.fine_tuned_models:
                return await self._generate_fine_tuned_quiz(text, key_concepts, num_questions, difficulty, text_type)
            else:
                return await self._generate_enhanced_quiz(text, key_concepts, num_questions, difficulty, text_type)
                
        except Exception as e:
            logger.error(f"Error generando quiz: {e}")
            return await super().generate_quiz(text, key_concepts, num_questions, difficulty)
    
    async def _generate_enhanced_quiz(self, text: str, key_concepts: List[str], 
                                    num_questions: int, difficulty: str, text_type: str) -> Dict[str, Any]:
        """Genera quiz mejorado según el tipo de texto"""
        try:
            questions = []
            
            # Extraer información especializada
            specialized_info = self._extract_specialized_info(text, text_type)
            
            # Crear preguntas específicas por tipo de texto
            if text_type == "historia":
                questions = self._create_history_questions(text, specialized_info, num_questions, difficulty)
            elif text_type == "ciencia":
                questions = self._create_science_questions(text, specialized_info, num_questions, difficulty)
            elif text_type == "tecnología":
                questions = self._create_tech_questions(text, specialized_info, num_questions, difficulty)
            else:
                questions = self._create_general_questions(text, key_concepts, num_questions, difficulty)
            
            # Completar con preguntas generales si es necesario
            while len(questions) < num_questions:
                additional_question = self._create_contextual_question(
                    len(questions) + 1, text, key_concepts, difficulty
                )
                questions.append(additional_question)
            
            return {
                "questions": questions[:num_questions],
                "success": True,
                "model_used": "enhanced_quiz_generator",
                "text_type": text_type
            }
            
        except Exception as e:
            logger.error(f"Error en quiz mejorado: {e}")
            return await super().generate_quiz(text, key_concepts, num_questions, difficulty)
    
    def _create_history_questions(self, text: str, info: Dict, num_questions: int, difficulty: str) -> List[Dict]:
        """Crea preguntas específicas para textos históricos"""
        questions = []
        
        # Preguntas sobre fechas y cronología
        if info.get("dates") and len(questions) < num_questions:
            dates = info["dates"]
            question = {
                "id": len(questions) + 1,
                "question": "¿Cuál es el período temporal principal abordado en el texto?",
                "options": [
                    f"Entre {dates[0]} y {dates[-1]}" if len(dates) > 1 else f"Durante {dates[0]}",
                    "Siglo XV",
                    "Era contemporánea",
                    "Prehistoria"
                ],
                "correct_answer": 0,
                "explanation": f"El texto se centra en el período {dates[0]} - {dates[-1] if len(dates) > 1 else 'mencionado'}, como se indica en las fechas principales del contenido.",
                "difficulty": difficulty
            }
            questions.append(question)
        
        # Preguntas sobre personajes históricos
        if info.get("people") and len(questions) < num_questions:
            people = info["people"]
            question = {
                "id": len(questions) + 1,
                "question": "¿Quién es una figura histórica importante mencionada en el texto?",
                "options": [
                    people[0],
                    "Napoleón Bonaparte",
                    "Julio César",
                    "Cleopatra"
                ],
                "correct_answer": 0,
                "explanation": f"{people[0]} es mencionado en el texto como una figura relevante para los eventos históricos descritos.",
                "difficulty": difficulty
            }
            questions.append(question)
        
        # Pregunta sobre causas y consecuencias
        if len(questions) < num_questions:
            question = {
                "id": len(questions) + 1,
                "question": "¿Cuál fue una de las principales consecuencias de los eventos descritos?",
                "options": [
                    "Transformación del orden político y social",
                    "Desaparición completa de las instituciones",
                    "Vuelta al sistema feudal",
                    "Eliminación de todas las fronteras"
                ],
                "correct_answer": 0,
                "explanation": "Los eventos históricos descritos tuvieron como consecuencia principal la transformación del orden político y social existente.",
                "difficulty": difficulty
            }
            questions.append(question)
        
        return questions
    
    def _create_science_questions(self, text: str, info: Dict, num_questions: int, difficulty: str) -> List[Dict]:
        """Crea preguntas específicas para textos científicos"""
        questions = []
        
        # Pregunta sobre procesos científicos
        if len(questions) < num_questions:
            question = {
                "id": len(questions) + 1,
                "question": "¿Cuál es el proceso principal descrito en el texto?",
                "options": [
                    "El proceso científico explicado en el contenido",
                    "Un proceso de manufactura industrial",
                    "Un proceso político",
                    "Un proceso puramente teórico sin aplicación"
                ],
                "correct_answer": 0,
                "explanation": "El texto describe un proceso científico específico con sus características y aplicaciones.",
                "difficulty": difficulty
            }
            questions.append(question)
        
        # Pregunta sobre aplicaciones
        if len(questions) < num_questions:
            question = {
                "id": len(questions) + 1,
                "question": "¿Cuál es la importancia práctica de lo descrito en el texto?",
                "options": [
                    "Tiene aplicaciones importantes en la vida cotidiana y la ciencia",
                    "Solo tiene valor teórico",
                    "Es completamente obsoleto",
                    "Solo se aplica en laboratorios especializados"
                ],
                "correct_answer": 0,
                "explanation": "Los conceptos científicos descritos tienen aplicaciones prácticas relevantes para la comprensión y mejora de procesos naturales o tecnológicos.",
                "difficulty": difficulty
            }
            questions.append(question)
        
        # Pregunta sobre características
        if len(questions) < num_questions:
            question = {
                "id": len(questions) + 1,
                "question": "¿Qué características fundamentales se destacan en el contenido científico?",
                "options": [
                    "Métodos basados en evidencia y experimentación",
                    "Creencias y tradiciones populares",
                    "Opiniones personales sin fundamento",
                    "Supersticiones y mitos antiguos"
                ],
                "correct_answer": 0,
                "explanation": "El contenido científico se caracteriza por estar basado en métodos rigurosos, evidencia empírica y experimentación controlada.",
                "difficulty": difficulty
            }
            questions.append(question)
        
        return questions
    
    def _create_tech_questions(self, text: str, info: Dict, num_questions: int, difficulty: str) -> List[Dict]:
        """Crea preguntas específicas para textos tecnológicos"""
        questions = []
        
        # Pregunta sobre funcionalidad
        if len(questions) < num_questions:
            question = {
                "id": len(questions) + 1,
                "question": "¿Cuál es la principal funcionalidad de la tecnología descrita?",
                "options": [
                    "Mejorar la eficiencia y capacidades en tareas específicas",
                    "Reemplazar completamente el trabajo humano",
                    "Crear problemas tecnológicos adicionales",
                    "Funcionar solo en condiciones ideales"
                ],
                "correct_answer": 0,
                "explanation": "Las tecnologías descritas tienen como objetivo principal mejorar la eficiencia y ampliar las capacidades humanas en diversas tareas.",
                "difficulty": difficulty
            }
            questions.append(question)
        
        # Pregunta sobre aplicaciones
        if len(questions) < num_questions:
            question = {
                "id": len(questions) + 1,
                "question": "¿En qué ámbitos se puede aplicar esta tecnología?",
                "options": [
                    "En múltiples sectores y disciplinas",
                    "Solo en investigación académica",
                    "Únicamente en empresas tecnológicas",
                    "Solo en el sector militar"
                ],
                "correct_answer": 0,
                "explanation": "La tecnología descrita tiene aplicaciones versátiles que se extienden a múltiples sectores y disciplinas.",
                "difficulty": difficulty
            }
            questions.append(question)
        
        return questions
    
    def _create_general_questions(self, text: str, key_concepts: List[str], num_questions: int, difficulty: str) -> List[Dict]:
        """Crea preguntas generales para cualquier tipo de texto"""
        questions = []
        
        # Pregunta sobre tema principal
        if len(questions) < num_questions:
            question = {
                "id": len(questions) + 1,
                "question": "¿Cuál es el tema central del texto analizado?",
                "options": [
                    "El tema principal desarrollado a lo largo del contenido",
                    "Un tema secundario mencionado brevemente",
                    "Información no relacionada con el contenido",
                    "Datos estadísticos sin contexto"
                ],
                "correct_answer": 0,
                "explanation": "El texto se centra en desarrollar un tema principal específico con información detallada y coherente.",
                "difficulty": difficulty
            }
            questions.append(question)
        
        # Pregunta sobre conceptos clave
        if key_concepts and len(questions) < num_questions:
            concept = key_concepts[0] if key_concepts else "el concepto principal"
            question = {
                "id": len(questions) + 1,
                "question": f"¿Qué papel juega {concept} en el contexto del texto?",
                "options": [
                    f"{concept} es un elemento fundamental para la comprensión del tema",
                    f"{concept} se menciona solo de forma tangencial",
                    f"{concept} contradice la información principal",
                    f"{concept} no tiene relación con el contenido"
                ],
                "correct_answer": 0,
                "explanation": f"{concept} representa un elemento clave que contribuye significativamente a la comprensión integral del tema tratado.",
                "difficulty": difficulty
            }
            questions.append(question)
        
        return questions
    
    def _create_contextual_question(self, question_id: int, text: str, concepts: List[str], difficulty: str) -> Dict:
        """Crea una pregunta contextual específica del texto"""
        
        # Extraer primera oración significativa
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 50]
        
        if meaningful_sentences:
            context_sentence = meaningful_sentences[0]
            
            return {
                "id": question_id,
                "question": "Según la información presentada en el texto, ¿cuál es la afirmación más precisa?",
                "options": [
                    context_sentence[:80] + "..." if len(context_sentence) > 80 else context_sentence,
                    "Una afirmación contradictoria al contenido del texto",
                    "Información no respaldada por el contenido analizado",
                    "Datos irrelevantes para el tema principal"
                ],
                "correct_answer": 0,
                "explanation": "Esta información proviene directamente del texto analizado y representa fielmente el contenido y contexto presentado.",
                "difficulty": difficulty
            }
        
        # Pregunta de respaldo
        return {
            "id": question_id,
            "question": "¿Cuál es el enfoque principal del contenido analizado?",
            "options": [
                "Proporcionar información educativa estructurada sobre el tema central",
                "Presentar datos sin conexión temática",
                "Contradecir información previamente establecida",
                "Ofrecer entretenimiento sin valor educativo"
            ],
            "correct_answer": 0,
            "explanation": "El texto mantiene un enfoque educativo claro, proporcionando información estructurada y coherente sobre el tema principal.",
            "difficulty": difficulty
        }
    
    async def generate_feedback(self, score: int, total: int, 
                              incorrect_questions: List[int], concepts: List[str]) -> str:
        """Genera feedback educativo personalizado y constructivo"""
        
        percentage = (score / total) * 100
        
        # Determinar tipo de contenido basado en conceptos
        text_type = self._infer_type_from_concepts(concepts)
        
        # Crear feedback estructurado y motivador
        feedback = self._create_comprehensive_feedback(score, total, percentage, concepts, incorrect_questions, text_type)
        
        return feedback
    
    def _infer_type_from_concepts(self, concepts: List[str]) -> str:
        """Infiere el tipo de contenido basado en los conceptos"""
        if not concepts:
            return "general"
        
        concept_text = " ".join(concepts).lower()
        
        for text_type, patterns in self.text_patterns.items():
            for keyword in patterns["keywords"]:
                if keyword in concept_text:
                    return text_type
        
        return "general"
    
    def _create_comprehensive_feedback(self, score: int, total: int, percentage: float, 
                                     concepts: List[str], incorrect_questions: List[int], text_type: str) -> str:
        """Crea feedback comprehensivo y personalizado"""
        
        # Encabezado motivacional según rendimiento
        if percentage >= 90:
            header = "🎉 **¡RENDIMIENTO EXCEPCIONAL!**"
            emoji = "🏆"
        elif percentage >= 80:
            header = "✨ **¡EXCELENTE TRABAJO!**"
            emoji = "🌟"
        elif percentage >= 70:
            header = "👍 **¡BUEN DESEMPEÑO!**"
            emoji = "📈"
        elif percentage >= 60:
            header = "💪 **¡PROGRESO POSITIVO!**"
            emoji = "🎯"
        else:
            header = "🌱 **¡OPORTUNIDAD DE CRECIMIENTO!**"
            emoji = "📚"
        
        feedback = f"{header}\n\n"
        
        # Resultado numérico destacado
        feedback += f"{emoji} **RESULTADO:** {score}/{total} respuestas correctas (**{percentage:.1f}%**)\n\n"
        
        # Análisis detallado por rango de rendimiento
        if percentage >= 90:
            feedback += self._create_excellent_feedback(concepts, text_type)
        elif percentage >= 70:
            feedback += self._create_good_feedback(concepts, incorrect_questions, text_type)
        elif percentage >= 50:
            feedback += self._create_improvement_feedback(concepts, text_type)
        else:
            feedback += self._create_foundational_feedback(concepts, text_type)
        
        # Agregar recomendaciones específicas por tipo de contenido
        feedback += self._add_content_specific_recommendations(text_type, percentage)
        
        return feedback
    
    def _create_excellent_feedback(self, concepts: List[str], text_type: str) -> str:
        """Feedback para rendimiento excepcional"""
        feedback = ("🔍 **ANÁLISIS DE RENDIMIENTO:**\n"
                   "Has demostrado un dominio sobresaliente del tema. Tu comprensión "
                   "de los conceptos clave es sólida y tu capacidad de análisis es ejemplar.\n\n")
        
        if concepts:
            feedback += f"💎 **FORTALEZAS IDENTIFICADAS:**\n"
            feedback += f"• Excelente manejo de {concepts[0]}\n"
            if len(concepts) > 1:
                feedback += f"• Comprensión avanzada de {concepts[1]}\n"
            feedback += "• Capacidad de síntesis y análisis crítico\n\n"
        
        feedback += ("🚀 **PRÓXIMOS DESAFÍOS:**\n"
                    "• Explora aspectos más profundos del tema\n"
                    "• Busca conexiones con temas relacionados\n"
                    "• Comparte tu conocimiento con otros estudiantes\n\n")
        
        return feedback
    
    def _create_good_feedback(self, concepts: List[str], incorrect_questions: List[int], text_type: str) -> str:
        """Feedback para buen rendimiento"""
        feedback = ("🔍 **ANÁLISIS DE RENDIMIENTO:**\n"
                   "Tienes una base sólida de conocimientos. Has captado los conceptos "
                   "principales, aunque hay algunas áreas específicas que puedes pulir.\n\n")
        
        if incorrect_questions:
            feedback += f"📊 **ÁREAS DE OPORTUNIDAD:**\n"
            feedback += f"• Revisar preguntas {', '.join(map(str, incorrect_questions[:3]))}\n"
            if concepts:
                feedback += f"• Profundizar en {concepts[-1] if len(concepts) > 1 else concepts[0]}\n"
            feedback += "\n"
        
        feedback += ("💡 **ESTRATEGIAS DE MEJORA:**\n"
                    "• Repasa los conceptos donde tuviste dificultades\n"
                    "• Busca ejemplos adicionales de los temas complejos\n"
                    "• Practica con ejercicios similares\n\n")
        
        return feedback
    
    def _create_improvement_feedback(self, concepts: List[str], text_type: str) -> str:
        """Feedback para rendimiento que necesita mejora"""
        feedback = ("🔍 **ANÁLISIS DE RENDIMIENTO:**\n"
                   "Estás construyendo una base de conocimientos sólida. Cada respuesta "
                   "correcta representa progreso real en tu comprensión del tema.\n\n")
        
        if concepts:
            feedback += f"🎯 **ENFOQUE RECOMENDADO:**\n"
            feedback += f"• Refuerza los fundamentos de {concepts[0]}\n"
            if len(concepts) > 1:
                feedback += f"• Practica más con {concepts[1]}\n"
            feedback += "\n"
        
        feedback += ("📖 **PLAN DE ESTUDIO:**\n"
                    "• Dedica tiempo a comprender los conceptos básicos\n"
                    "• Utiliza recursos adicionales como videos o diagramas\n"
                    "• Practica con ejercicios de dificultad gradual\n\n")
        
        return feedback
    
    def _create_foundational_feedback(self, concepts: List[str], text_type: str) -> str:
        """Feedback para rendimiento que necesita refuerzo fundamental"""
        feedback = ("🔍 **ANÁLISIS DE RENDIMIENTO:**\n"
                   "Estás en proceso de construcción de conocimientos fundamentales. "
                   "No te desanimes, cada intento es una oportunidad valiosa de aprendizaje.\n\n")
        
        if concepts:
            feedback += f"🎯 **ENFOQUE FUNDAMENTAL:**\n"
            feedback += f"• Comienza con los conceptos básicos de {concepts[0]}\n"
            feedback += f"• Dedica tiempo extra a entender definiciones clave\n\n"
        
        feedback += ("📖 **ESTRATEGIA DE APRENDIZAJE:**\n"
                    "• Estudia un concepto a la vez hasta dominarlo\n"
                    "• Usa analogías y ejemplos cotidianos\n"
                    "• Practica con ejercicios muy básicos primero\n"
                    "• Busca ayuda cuando la necesites\n\n")
        
        feedback += ("🌟 **MENSAJE MOTIVACIONAL:**\n"
                    "El aprendizaje es un proceso gradual y personal. Cada paso que das "
                    "te acerca más al dominio del tema. Tu perseverancia es clave para el éxito.\n\n")
        
        return feedback
    
    def _add_content_specific_recommendations(self, text_type: str, percentage: float) -> str:
        """Agrega recomendaciones específicas según el tipo de contenido"""
        
        recommendations = "🎓 **RECOMENDACIONES ESPECÍFICAS:**\n"
        
        if text_type == "historia":
            if percentage >= 80:
                recommendations += ("• Explora fuentes primarias del período estudiado\n"
                                  "• Analiza diferentes perspectivas históricas\n"
                                  "• Conecta eventos con consecuencias a largo plazo\n")
            else:
                recommendations += ("• Crea líneas de tiempo para organizar eventos\n"
                                  "• Estudia mapas históricos del período\n"
                                  "• Memoriza fechas clave y personajes importantes\n")
        
        elif text_type == "ciencia":
            if percentage >= 80:
                recommendations += ("• Busca aplicaciones prácticas de los conceptos\n"
                                  "• Realiza experimentos relacionados si es posible\n"
                                  "• Investiga avances recientes en el área\n")
            else:
                recommendations += ("• Repasa las leyes y principios fundamentales\n"
                                  "• Practica con diagramas y esquemas\n"
                                  "• Relaciona conceptos con ejemplos cotidianos\n")
        
        elif text_type == "tecnología":
            if percentage >= 80:
                recommendations += ("• Experimenta con herramientas relacionadas\n"
                                  "• Sigue las tendencias tecnológicas actuales\n"
                                  "• Considera aplicaciones innovadoras\n")
            else:
                recommendations += ("• Familiarízate con la terminología básica\n"
                                  "• Comprende los fundamentos antes de avanzar\n"
                                  "• Practica con ejemplos step-by-step\n")
        
        else:
            recommendations += ("• Refuerza la comprensión lectora\n"
                              "• Practica el análisis de textos similares\n"
                              "• Desarrolla técnicas de estudio efectivas\n")
        
        return recommendations
    
    def get_model_status(self) -> Dict[str, Any]:
        """Obtiene el estado completo de todos los modelos mejorados"""
        
        status = {
            "base_models": {
                "summarizer_loaded": self.summarizer is not None,
                "t5_model_loaded": self.t5_model is not None,
                "classifier_loaded": self.classifier is not None
            },
            "enhanced_models": {
                "enhanced_summarizer": hasattr(self, 'enhanced_summarizer'),
                "enhanced_generator": hasattr(self, 'enhanced_generator')
            },
            "fine_tuned_models": {
                "summarizer_loaded": "summarizer" in self.fine_tuned_models,
                "question_gen_loaded": "question_generator" in self.fine_tuned_models,
                "feedback_gen_loaded": "feedback_generator" in self.fine_tuned_models
            },
            "universal_patterns": {
                "text_types_supported": list(self.text_patterns.keys()),
                "pattern_count": len(self.text_patterns)
            },
            "device": self.device,
            "model_config_loaded": self.model_config is not None,
            "enhanced_features": True,
            "universal_text_support": True
        }
        
        if self.model_config:
            status["training_info"] = self.model_config.get("training_info", {})
        
        return status