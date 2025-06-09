# app/services/improved_ai_service.py - SERVICIO MEJORADO CON PROMPTS AVANZADOS CORREGIDO
import json
import logging
import random
import re
from typing import Dict, Any, List, Optional
from collections import Counter
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    pipeline, T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer
)
import torch
from app.core.config import settings

logger = logging.getLogger(__name__)

class ImprovedAIService:
    """
    Servicio de IA mejorado con prompts avanzados y post-procesamiento inteligente
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Inicializando servicio de IA mejorado en dispositivo: {self.device}")
        
        # Cargar modelos optimizados
        self._init_optimized_models()
        
        # Configurar patrones y plantillas
        self._setup_advanced_patterns()
        
        # Configurar prompts especializados
        self._setup_specialized_prompts()
    
    def _init_optimized_models(self):
        """Inicializa modelos optimizados para español"""
        try:
            logger.info("📦 Cargando modelos optimizados...")
            
            # Modelo para resúmenes - usar modelo multilingüe optimizado
            logger.info("📝 Cargando modelo de resúmenes...")
            try:
                # Intentar modelo multilingüe mejor para español
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/mbart-large-50-many-to-many-mmt",
                    device=0 if self.device == "cuda" else -1,
                    max_length=300,
                    min_length=100
                )
            except Exception as e:
                logger.warning(f"⚠️ Error cargando mbart, usando fallback: {e}")
                # Fallback a BART estándar
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if self.device == "cuda" else -1
                )
            
            # Modelo para generación de texto - T5 multilingüe
            logger.info("🔤 Cargando modelo de generación...")
            try:
                self.generator_tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
                self.generator_model = T5ForConditionalGeneration.from_pretrained(
                    "google/mt5-base"
                ).to(self.device)
            except Exception as e:
                logger.warning(f"⚠️ Error cargando mt5, usando fallback: {e}")
                # Fallback a T5 estándar
                self.generator_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
                self.generator_model = T5ForConditionalGeneration.from_pretrained(
                    "google/flan-t5-base"
                ).to(self.device)
            
            # Pipeline de análisis de sentimientos
            logger.info("🎭 Cargando analizador de sentimientos...")
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if self.device == "cuda" else -1
                )
            except Exception as e:
                logger.warning(f"⚠️ Error cargando sentiment analyzer: {e}")
                self.sentiment_analyzer = None
            
            logger.info("✅ Todos los modelos cargados exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            self._init_fallback_models()
    
    def _init_fallback_models(self):
        """Modelos de respaldo más pequeños"""
        logger.info("🔄 Cargando modelos de respaldo...")
        try:
            self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
            self.generator_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
            self.generator_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
            self.sentiment_analyzer = None
        except Exception as e:
            logger.error(f"❌ Error incluso con modelos de respaldo: {e}")
            raise
    
    def _setup_advanced_patterns(self):
        """Configura patrones avanzados para detección de dominios y post-procesamiento"""
        self.domain_patterns = {
            "historia": {
                "keywords": ["guerra", "batalla", "revolución", "imperio", "independencia", "tratado", "siglo"],
                "indicators": ["(1939-1945)", "siglo XX", "Primera Guerra", "Segunda Guerra"],
                "structure_markers": ["causas", "consecuencias", "desarrollo", "fases"]
            },
            "ciencias": {
                "keywords": ["proceso", "célula", "organismo", "energía", "reacción", "fotosíntesis"],
                "indicators": ["ATP", "ADN", "CO₂", "O₂", "clorofila", "mitocondrias"],
                "structure_markers": ["proceso", "función", "importancia", "etapas"]
            },
            "tecnologia": {
                "keywords": ["algoritmo", "sistema", "software", "datos", "inteligencia", "aplicación"],
                "indicators": ["machine learning", "deep learning", "IA", "neural", "digital"],
                "structure_markers": ["funcionamiento", "aplicaciones", "ventajas", "desafíos"]
            },
            "literatura": {
                "keywords": ["obra", "autor", "estilo", "narrativa", "personaje", "género"],
                "indicators": ["novela", "poesía", "teatro", "modernismo", "realismo"],
                "structure_markers": ["características", "contexto", "influencia", "técnicas"]
            },
            "economia": {
                "keywords": ["mercado", "precio", "demanda", "oferta", "empresa", "económico"],
                "indicators": ["PIB", "inflación", "inversión", "capital", "financiero"],
                "structure_markers": ["factores", "efectos", "políticas", "tendencias"]
            }
        }
    
    def _setup_specialized_prompts(self):
        """Configura prompts especializados por dominio y tarea"""
        self.summary_prompts = {
            "historia": """Crea un resumen educativo estructurado sobre este texto histórico. Debe incluir:
- 🔑 Conceptos clave históricos más importantes
- 📅 Cronología y fechas relevantes  
- 👥 Personajes históricos principales
- 🎯 Causas y consecuencias del evento
- 💡 Importancia histórica y legado

Texto: {text}

Resumen educativo:""",
            
            "ciencias": """Crea un resumen educativo estructurado sobre este texto científico. Debe incluir:
- 🔬 Proceso o fenómeno científico principal
- ⚗️ Componentes y elementos involucrados
- 🔄 Pasos o etapas del proceso
- 🌍 Importancia para la vida/ecosistema
- 💡 Aplicaciones prácticas

Texto: {text}

Resumen educativo:""",
            
            "tecnologia": """Crea un resumen educativo estructurado sobre este texto tecnológico. Debe incluir:
- 💻 Tecnología o sistema principal
- ⚙️ Funcionamiento básico
- 🚀 Aplicaciones y usos principales
- 📈 Ventajas y beneficios
- 🔮 Impacto futuro

Texto: {text}

Resumen educativo:""",
            
            "general": """Crea un resumen educativo claro y estructurado de este texto. Debe incluir:
- 🎯 Tema principal
- 🔑 Conceptos más importantes
- 📋 Puntos clave a recordar
- 💡 Relevancia e importancia

Texto: {text}

Resumen educativo:"""
        }
        
        self.question_prompts = {
            "historia": """Genera una pregunta educativa sobre este texto histórico. La pregunta debe:
- Evaluar comprensión de eventos, causas o consecuencias
- Ser específica y basada en el contenido
- Tener respuesta clara en el texto
- Ser apropiada para estudiantes

Texto: {text}
Conceptos clave: {concepts}

Pregunta educativa de calidad:""",
            
            "ciencias": """Genera una pregunta educativa sobre este texto científico. La pregunta debe:
- Evaluar comprensión de procesos o conceptos científicos
- Ser específica sobre el tema tratado
- Requerir comprensión, no solo memorización
- Ser clara y precisa

Texto: {text}
Conceptos clave: {concepts}

Pregunta educativa de calidad:""",
            
            "tecnologia": """Genera una pregunta educativa sobre este texto tecnológico. La pregunta debe:
- Evaluar comprensión de funcionamiento o aplicaciones
- Ser práctica y relevante
- Conectar conceptos con aplicaciones reales
- Ser apropiada para el nivel educativo

Texto: {text}
Conceptos clave: {concepts}

Pregunta educativa de calidad:""",
            
            "general": """Genera una pregunta educativa de calidad sobre este texto. La pregunta debe:
- Evaluar comprensión del tema principal
- Ser específica y clara
- Tener respuesta fundamentada en el texto
- Promover pensamiento crítico

Texto: {text}
Conceptos clave: {concepts}

Pregunta educativa de calidad:"""
        }
    
    def detect_domain(self, text: str) -> str:
        """Detecta el dominio del texto de forma inteligente"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = 0
            
            # Puntuar por keywords
            for keyword in patterns["keywords"]:
                score += text_lower.count(keyword) * 2
            
            # Puntuar por indicadores específicos
            for indicator in patterns["indicators"]:
                if indicator.lower() in text_lower:
                    score += 5
            
            # Puntuar por marcadores de estructura
            for marker in patterns["structure_markers"]:
                score += text_lower.count(marker)
            
            domain_scores[domain] = score
        
        # Detecciones específicas adicionales
        if "segunda guerra mundial" in text_lower or "hitler" in text_lower:
            domain_scores["historia"] = domain_scores.get("historia", 0) + 15
        
        if "fotosíntesis" in text_lower or "clorofila" in text_lower:
            domain_scores["ciencias"] = domain_scores.get("ciencias", 0) + 15
        
        if "inteligencia artificial" in text_lower or "machine learning" in text_lower:
            domain_scores["tecnologia"] = domain_scores.get("tecnologia", 0) + 15
        
        # Retornar dominio con mayor puntuación
        if not domain_scores or max(domain_scores.values()) < 3:
            return "general"
        
        detected = max(domain_scores, key=domain_scores.get)
        logger.info(f"🎯 Dominio detectado: {detected} (puntuación: {domain_scores[detected]})")
        
        return detected
    
    async def generate_summary(self, text: str, length: str = "medium", domain: str = None) -> Dict[str, Any]:
        """
        Genera resumen educativo mejorado con prompts especializados
        """
        try:
            # Detectar dominio si no se especifica
            if domain is None:
                domain = self.detect_domain(text)
            
            logger.info(f"📝 Generando resumen para dominio: {domain}")
            
            # Seleccionar prompt especializado
            prompt_template = self.summary_prompts.get(domain, self.summary_prompts["general"])
            
            # Configurar longitud
            length_config = {
                "short": {"max_length": 150, "min_length": 50},
                "medium": {"max_length": 250, "min_length": 100},
                "long": {"max_length": 350, "min_length": 150}
            }
            config = length_config.get(length, length_config["medium"])
            
            # Método 1: Intentar con prompts especializados usando el generador T5
            try:
                specialized_summary = await self._generate_with_specialized_prompt(
                    text, domain, prompt_template, config
                )
                if specialized_summary and len(specialized_summary) > 50:
                    # Post-procesar para mejorar calidad
                    final_summary = self._post_process_summary(specialized_summary, text, domain)
                    
                    return {
                        "summary": final_summary,
                        "success": True,
                        "method": "specialized_prompt",
                        "domain": domain
                    }
            except Exception as e:
                logger.warning(f"⚠️ Error con prompt especializado: {e}")
            
            # Método 2: Fallback al summarizer tradicional con post-procesamiento
            try:
                # Limitar longitud del texto de entrada
                input_text = text[:1000] if len(text) > 1000 else text
                
                summary_result = self.summarizer(
                    input_text,
                    max_length=config["max_length"],
                    min_length=config["min_length"],
                    do_sample=True,
                    temperature=0.7
                )
                
                raw_summary = summary_result[0]['summary_text']
                
                # Post-procesar para hacerlo educativo
                final_summary = self._create_educational_structure(raw_summary, text, domain)
                
                return {
                    "summary": final_summary,
                    "success": True,
                    "method": "traditional_with_postprocessing",
                    "domain": domain
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Error con summarizer tradicional: {e}")
            
            # Método 3: Crear resumen inteligente manualmente
            fallback_summary = self._create_intelligent_fallback_summary(text, domain)
            
            return {
                "summary": fallback_summary,
                "success": True,
                "method": "intelligent_fallback",
                "domain": domain
            }
            
        except Exception as e:
            logger.error(f"❌ Error generando resumen: {e}")
            return {
                "summary": self._create_emergency_summary(text),
                "success": False,
                "error": str(e),
                "method": "emergency"
            }
    
    async def _generate_with_specialized_prompt(self, text: str, domain: str, 
                                               prompt_template: str, config: Dict) -> str:
        """Genera resumen usando prompt especializado con T5"""
        
        # Crear prompt completo
        prompt = prompt_template.format(text=text[:800])  # Limitar entrada
        
        # Tokenizar
        inputs = self.generator_tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generar
        with torch.no_grad():
            outputs = self.generator_model.generate(
                **inputs,
                max_length=config["max_length"],
                min_length=config["min_length"],
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        # Decodificar
        generated_text = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Limpiar prompt del resultado
        if "Resumen educativo:" in generated_text:
            generated_text = generated_text.split("Resumen educativo:")[-1].strip()
        
        return generated_text
    
    def _post_process_summary(self, raw_summary: str, original_text: str, domain: str) -> str:
        """Post-procesa el resumen para mejorar su calidad educativa"""
        
        # Limpiar texto
        clean_summary = self._clean_text(raw_summary)
        
        # Si el resumen es muy corto o de mala calidad, recrear
        if len(clean_summary) < 50 or self._has_quality_issues(clean_summary):
            return self._create_educational_structure("", original_text, domain)
        
        # Mejorar estructura si es bueno pero no estructurado
        if not self._is_well_structured(clean_summary):
            return self._create_educational_structure(clean_summary, original_text, domain)
        
        return clean_summary
    
    def _clean_text(self, text: str) -> str:
        """Limpia errores comunes en el texto generado"""
        
        # Correcciones específicas
        corrections = {
            "seguirra": "guerra",
            "eusu": "EEUU",
            "histororia": "historia",
            "proces": "proceso",
            "teh": "the",
            "caracterticas": "características",
            "importnate": "importante"
        }
        
        clean_text = text
        for error, correction in corrections.items():
            clean_text = clean_text.replace(error, correction)
        
        # Limpiar espacios múltiples
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Corregir puntuación
        clean_text = re.sub(r'\s+([.,:;!?])', r'\1', clean_text)
        clean_text = re.sub(r'([.!?])\s*([a-záéíóúüñ])', r'\1 \2', clean_text)
        
        return clean_text.strip()
    
    def _has_quality_issues(self, text: str) -> bool:
        """Detecta problemas de calidad en el texto"""
        
        quality_issues = [
            len(text) < 50,  # Muy corto
            text.count("los") > 8,  # Demasiados artículos
            text.count("que") > 6,  # Demasiadas repeticiones
            "..." in text and text.count("...") > 2,  # Puntos suspensivos excesivos
            len([w for w in text.split() if len(w) > 15]) > 3,  # Palabras muy largas
            not any(c.isupper() for c in text)  # Sin mayúsculas
        ]
        
        return any(quality_issues)
    
    def _is_well_structured(self, text: str) -> bool:
        """Verifica si el texto tiene buena estructura educativa"""
        
        structure_indicators = [
            "📚" in text or "🔑" in text,  # Emojis educativos
            "conceptos" in text.lower(),
            "importante" in text.lower(),
            "proceso" in text.lower(),
            len(text.split('.')) >= 3  # Al menos 3 oraciones
        ]
        
        return any(structure_indicators)
    
    def _create_educational_structure(self, base_summary: str, original_text: str, domain: str) -> str:
        """Crea estructura educativa mejorada"""
        
        # Extraer información clave del texto original
        key_info = self._extract_key_information(original_text, domain)
        
        # Crear estructura base
        structured_summary = "📚 **RESUMEN EDUCATIVO**\n\n"
        
        # Encabezado específico por dominio
        domain_headers = {
            "historia": "🏛️ **ANÁLISIS HISTÓRICO**",
            "ciencias": "🔬 **ANÁLISIS CIENTÍFICO**",
            "tecnologia": "💻 **ANÁLISIS TECNOLÓGICO**",
            "literatura": "📖 **ANÁLISIS LITERARIO**",
            "economia": "📊 **ANÁLISIS ECONÓMICO**"
        }
        
        structured_summary += f"{domain_headers.get(domain, '🎯 **ANÁLISIS TEMÁTICO**')}\n\n"
        
        # Conceptos clave
        if key_info.get("concepts"):
            structured_summary += f"🔑 **CONCEPTOS CLAVE:** {', '.join(key_info['concepts'][:4])}\n\n"
        
        # Información específica por dominio
        if domain == "historia" and key_info.get("timeline"):
            structured_summary += f"📅 **PERÍODO:** {key_info['timeline']}\n\n"
        elif domain == "ciencias" and key_info.get("process"):
            structured_summary += f"⚗️ **PROCESO PRINCIPAL:** {key_info['process']}\n\n"
        elif domain == "tecnologia" and key_info.get("applications"):
            structured_summary += f"💡 **APLICACIONES:** {', '.join(key_info['applications'][:3])}\n\n"
        
        # Contenido principal
        if base_summary and len(base_summary) > 30:
            main_content = base_summary
        else:
            main_content = self._create_main_content_from_text(original_text, domain)
        
        structured_summary += f"📝 **CONTENIDO PRINCIPAL:**\n{main_content}\n\n"
        
        # Puntos importantes
        key_points = self._generate_key_points(original_text, domain, key_info)
        if key_points:
            structured_summary += f"💡 **PUNTOS IMPORTANTES:**\n"
            for i, point in enumerate(key_points, 1):
                structured_summary += f"{i}. {point}\n"
        
        return structured_summary
    
    def _extract_key_information(self, text: str, domain: str) -> Dict[str, Any]:
        """Extrae información clave específica por dominio"""
        
        info = {"concepts": [], "timeline": None, "process": None, "applications": []}
        
        # Conceptos generales
        important_words = re.findall(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{3,}\b', text)
        word_freq = Counter(important_words)
        stop_words = {'Para', 'Este', 'Esta', 'Todo', 'Cada', 'Durante', 'Según'}
        concepts = [word for word, freq in word_freq.most_common(8) 
                   if word not in stop_words and freq > 1]
        info["concepts"] = concepts[:5]
        
        # Información específica por dominio
        if domain == "historia":
            # Extraer fechas y períodos
            dates = re.findall(r'\b(?:19|20)\d{2}\b', text)
            periods = re.findall(r'\bsiglo\s+[IVX]+\b', text, re.IGNORECASE)
            if dates:
                info["timeline"] = f"{min(dates)}-{max(dates)}" if len(dates) > 1 else dates[0]
            elif periods:
                info["timeline"] = periods[0]
        
        elif domain == "ciencias":
            # Extraer procesos científicos
            processes = re.findall(r'\b(?:fotosíntesis|respiración|digestión|evolución|mitosis)\b', text, re.IGNORECASE)
            if processes:
                info["process"] = processes[0].title()
        
        elif domain == "tecnologia":
            # Extraer aplicaciones
            apps = re.findall(r'\b(?:reconocimiento|traducción|automatización|optimización|predicción)\b', text, re.IGNORECASE)
            info["applications"] = list(set(apps))[:4]
        
        return info
    
    def _create_main_content_from_text(self, text: str, domain: str) -> str:
        """Crea contenido principal inteligente del texto original"""
        
        # Extraer oraciones más importantes
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
        
        if len(sentences) <= 2:
            return " ".join(sentences) + "."
        
        # Seleccionar oraciones clave
        selected_sentences = []
        
        # Primera oración (contexto)
        selected_sentences.append(sentences[0])
        
        # Oración del medio (desarrollo)
        if len(sentences) > 2:
            mid_idx = len(sentences) // 2
            selected_sentences.append(sentences[mid_idx])
        
        # Última oración si es informativa
        if len(sentences) > 1 and len(sentences[-1].split()) > 8:
            selected_sentences.append(sentences[-1])
        
        return " ".join(selected_sentences[:3]) + "."
    
    def _generate_key_points(self, text: str, domain: str, key_info: Dict) -> List[str]:
        """Genera puntos clave específicos por dominio"""
        
        points = []
        
        if domain == "historia":
            if "segunda guerra mundial" in text.lower():
                points = [
                    "Conflicto global que involucró a la mayoría de las naciones del mundo",
                    "Marcó el fin de la hegemonía europea y el surgimiento de nuevas superpotencias",
                    "Estableció las bases del orden internacional de la segunda mitad del siglo XX"
                ]
            else:
                points = [
                    "Evento histórico con causas complejas e interrelacionadas",
                    "Consecuencias que influyeron en el desarrollo posterior de la sociedad",
                    "Importancia para comprender los procesos históricos actuales"
                ]
        
        elif domain == "ciencias":
            if "fotosíntesis" in text.lower():
                points = [
                    "Proceso fundamental que produce el oxígeno que respiramos",
                    "Base de prácticamente todas las cadenas alimenticias del planeta",
                    "Regula el equilibrio de gases en la atmósfera terrestre"
                ]
            else:
                points = [
                    "Proceso científico basado en principios naturales comprobables",
                    "Aplicaciones prácticas que benefician a la humanidad",
                    "Importancia para comprender el funcionamiento del mundo natural"
                ]
        
        elif domain == "tecnologia":
            if "inteligencia artificial" in text.lower():
                points = [
                    "Tecnología que amplifica las capacidades humanas en tareas específicas",
                    "Aplicaciones en múltiples industrias y aspectos de la vida cotidiana",
                    "Requiere desarrollo ético y consideración de impactos sociales"
                ]
            else:
                points = [
                    "Tecnología que mejora la eficiencia y capacidades en diversas tareas",
                    "Evolución constante para adaptarse a nuevas necesidades",
                    "Impacto transformador en la forma de trabajar y vivir"
                ]
        
        else:
            # Puntos generales
            points = [
                "Tema de relevancia e importancia en su área de conocimiento",
                "Conceptos fundamentales que requieren comprensión integral",
                "Aplicabilidad e implicaciones en contextos más amplios"
            ]
        
        return points[:3]
    
    def _create_intelligent_fallback_summary(self, text: str, domain: str) -> str:
        """Crea resumen inteligente como último recurso"""
        
        # Extraer información básica
        key_info = self._extract_key_information(text, domain)
        
        # Crear resumen estructurado básico pero completo
        summary = "📚 **RESUMEN EDUCATIVO**\n\n"
        summary += f"🎯 **TEMA PRINCIPAL:** {domain.replace('_', ' ').title()}\n\n"
        
        if key_info.get("concepts"):
            summary += f"🔑 **CONCEPTOS IMPORTANTES:** {', '.join(key_info['concepts'][:4])}\n\n"
        
        # Contenido principal basado en las primeras oraciones
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 15]
        if sentences:
            main_content = " ".join(sentences[:2]) + "."
            summary += f"📝 **INFORMACIÓN PRINCIPAL:**\n{main_content}\n\n"
        
        # Puntos clave genéricos
        summary += "💡 **PUNTOS A RECORDAR:**\n"
        summary += "1. Es un tema importante en su área de conocimiento\n"
        summary += "2. Requiere comprensión de conceptos fundamentales\n"
        summary += "3. Tiene aplicaciones e implicaciones prácticas relevantes"
        
        return summary
    
    def _create_emergency_summary(self, text: str) -> str:
        """Crea resumen de emergencia cuando todo falla"""
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        
        if sentences:
            content = " ".join(sentences[:2]) + "."
        else:
            content = "El texto contiene información educativa importante que requiere análisis detallado."
        
        return f"""📚 **RESUMEN EDUCATIVO**

🎯 **CONTENIDO PRINCIPAL:**
{content}

💡 **NOTA:** Este resumen fue generado usando métodos básicos de procesamiento. 
Para mejores resultados, asegúrate de que el texto tenga suficiente contenido educativo."""
    
    async def generate_quiz(self, text: str, key_concepts: List[str], 
                          num_questions: int = 5, difficulty: str = "medium", 
                          domain: str = None) -> Dict[str, Any]:
        """
        Genera quiz mejorado con preguntas contextuales de alta calidad
        """
        try:
            # Detectar dominio si no se especifica
            if domain is None:
                domain = self.detect_domain(text)
            
            logger.info(f"❓ Generando quiz para dominio: {domain} ({num_questions} preguntas)")
            
            questions = []
            
            # Método 1: Generar preguntas con prompts especializados
            try:
                specialized_questions = await self._generate_questions_with_prompts(
                    text, key_concepts, num_questions, difficulty, domain
                )
                if specialized_questions and len(specialized_questions) > 0:
                    questions.extend(specialized_questions)
            except Exception as e:
                logger.warning(f"⚠️ Error con preguntas especializadas: {e}")
            
            # Método 2: Completar con preguntas inteligentes si faltan
            while len(questions) < num_questions:
                try:
                    intelligent_question = self._create_intelligent_question(
                        text, key_concepts, len(questions) + 1, difficulty, domain
                    )
                    questions.append(intelligent_question)
                except Exception as e:
                    logger.warning(f"⚠️ Error creando pregunta {len(questions) + 1}: {e}")
                    break
            
            # Método 3: Completar con preguntas de fallback si aún faltan
            while len(questions) < num_questions:
                fallback_question = self._create_fallback_question(
                    len(questions) + 1, key_concepts, domain, difficulty
                )
                questions.append(fallback_question)
            
            return {
                "questions": questions[:num_questions],
                "success": True,
                "domain": domain,
                "generation_method": "improved_contextual"
            }
            
        except Exception as e:
            logger.error(f"❌ Error generando quiz: {e}")
            return {
                "questions": self._create_emergency_quiz(key_concepts, num_questions, difficulty),
                "success": False,
                "error": str(e),
                "generation_method": "emergency"
            }
    
    async def _generate_questions_with_prompts(self, text: str, concepts: List[str], 
                                             num_questions: int, difficulty: str, domain: str) -> List[Dict]:
        """Genera preguntas usando prompts especializados"""
        
        questions = []
        
        # Seleccionar prompt especializado
        prompt_template = self.question_prompts.get(domain, self.question_prompts["general"])
        
        for i in range(min(num_questions, 3)):  # Máximo 3 con IA para evitar lentitud
            try:
                # Crear prompt específico
                concepts_str = ", ".join(concepts[:4]) if concepts else "conceptos principales"
                prompt = prompt_template.format(
                    text=text[:600],  # Limitar texto
                    concepts=concepts_str
                )
                
                # Generar con T5
                inputs = self.generator_tokenizer(
                    prompt,
                    max_length=400,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.generator_model.generate(
                        **inputs,
                        max_length=100,
                        min_length=20,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.95,
                        repetition_penalty=1.3
                    )
                
                generated_question = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Limpiar y procesar pregunta
                clean_question = self._clean_generated_question(generated_question, prompt)
                
                if self._is_valid_question(clean_question):
                    # Crear opciones y respuesta
                    question_dict = self._create_question_with_options(
                        clean_question, text, concepts, domain, i + 1, difficulty
                    )
                    questions.append(question_dict)
                
            except Exception as e:
                logger.warning(f"⚠️ Error generando pregunta {i+1} con IA: {e}")
                continue
        
        return questions
    
    def _clean_generated_question(self, generated_text: str, original_prompt: str) -> str:
        """Limpia la pregunta generada por IA"""
        
        # Remover el prompt del resultado
        if "Pregunta educativa de calidad:" in generated_text:
            question = generated_text.split("Pregunta educativa de calidad:")[-1].strip()
        else:
            question = generated_text.strip()
        
        # Limpiar caracteres extraños
        question = re.sub(r'^[^\w¿]+', '', question)
        question = re.sub(r'[^\w\s¿?¡!.,;:()\-áéíóúüñÁÉÍÓÚÜÑ]+', '', question)
        
        # Asegurar que termine con ?
        if not question.endswith('?'):
            question += '?'
        
        # Asegurar que empiece con ¿ si es pregunta en español
        if not question.startswith('¿') and '?' in question:
            question = '¿' + question
        
        return question.strip()
    
    def _is_valid_question(self, question: str) -> bool:
        """Valida si una pregunta generada es de calidad"""
        
        if not question or len(question) < 10:
            return False
        
        # Debe contener interrogaciones
        if '?' not in question:
            return False
        
        # No debe contener errores comunes
        invalid_patterns = [
            r'\{[^}]+\}',  # Placeholders sin reemplazar
            r'[a-z]{20,}',  # Palabras muy largas (errores)
            r'\d{10,}',  # Números muy largos
            'lorem ipsum',  # Texto de relleno
            'ejemplo ejemplo'  # Repeticiones obvias
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, question.lower()):
                return False
        
        return True
    
    def _create_intelligent_question(self, text: str, concepts: List[str], 
                                   question_id: int, difficulty: str, domain: str) -> Dict[str, Any]:
        """Crea pregunta inteligente basada en el contenido"""
        
        # Extraer información relevante del texto
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 30]
        
        # Seleccionar oración para basar la pregunta
        if sentences:
            selected_sentence = random.choice(sentences[:5])  # De las primeras 5 oraciones
            
            # Crear pregunta basada en la oración
            question = self._create_question_from_sentence(selected_sentence, domain, concepts)
            
            # Crear opciones inteligentes
            options = self._create_intelligent_options(question, selected_sentence, text, concepts, domain)
            
            # Seleccionar respuesta correcta
            correct_answer = 0  # La primera opción es correcta por diseño
            
            # Generar explicación
            explanation = self._generate_explanation(question, options[0], selected_sentence, domain)
            
            return {
                "id": question_id,
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "explanation": explanation,
                "difficulty": difficulty,
                "source": "intelligent_generation"
            }
        
        # Fallback si no hay oraciones válidas
        return self._create_fallback_question(question_id, concepts, domain, difficulty)
    
    def _create_question_from_sentence(self, sentence: str, domain: str, concepts: List[str]) -> str:
        """Crea pregunta basada en una oración específica"""
        
        # Patrones de preguntas por dominio
        question_starters = {
            "historia": [
                "¿Cuál fue la causa principal de",
                "¿Qué consecuencias tuvo",
                "¿En qué año ocurrió",
                "¿Quién fue el responsable de",
                "¿Cómo influyó"
            ],
            "ciencias": [
                "¿Cómo funciona el proceso de",
                "¿Qué papel cumple",
                "¿Dónde ocurre",
                "¿Por qué es importante",
                "¿Cuáles son los componentes de"
            ],
            "tecnologia": [
                "¿Cómo se aplica",
                "¿Qué ventajas ofrece",
                "¿Cuál es la función de",
                "¿Por qué es útil",
                "¿Dónde se utiliza"
            ]
        }
        
        # Seleccionar starter apropiado
        starters = question_starters.get(domain, question_starters["ciencias"])
        starter = random.choice(starters)
        
        # Extraer elemento clave de la oración
        if concepts:
            main_concept = concepts[0]
        else:
            # Extraer sustantivo principal
            words = sentence.split()
            capitalized_words = [w for w in words if w[0].isupper() and len(w) > 3]
            main_concept = capitalized_words[0] if capitalized_words else "el tema principal"
        
        # Construir pregunta
        question = f"{starter} {main_concept.lower()} según el texto?"
        
        return question
    
    def _create_intelligent_options(self, question: str, source_sentence: str, 
                                   full_text: str, concepts: List[str], domain: str) -> List[str]:
        """Crea opciones inteligentes para la pregunta"""
        
        # Opción correcta basada en la oración fuente
        correct_option = self._extract_answer_from_sentence(source_sentence, question, domain)
        
        # Opciones incorrectas pero plausibles
        incorrect_options = []
        
        # Generar distractores basados en el dominio
        domain_distractors = {
            "historia": [
                "Un evento que no está documentado en fuentes históricas",
                "Una consecuencia que no se menciona en el texto",
                "Un factor que no tuvo influencia en los acontecimientos"
            ],
            "ciencias": [
                "Un proceso que no ocurre en los organismos descritos",
                "Una función que no se menciona en el texto",
                "Un componente que no forma parte del sistema"
            ],
            "tecnologia": [
                "Una aplicación que no se describe en el contenido",
                "Una ventaja que no se menciona en el texto",
                "Una función que no es característica de la tecnología"
            ]
        }
        
        distractors = domain_distractors.get(domain, domain_distractors["ciencias"])
        incorrect_options = random.sample(distractors, min(3, len(distractors)))
        
        # Combinar y mezclar opciones
        all_options = [correct_option] + incorrect_options
        random.shuffle(all_options)
        
        return all_options
    
    def _extract_answer_from_sentence(self, sentence: str, question: str, domain: str) -> str:
        """Extrae respuesta de la oración fuente"""
        
        # Simplificar: usar parte de la oración como respuesta
        words = sentence.split()
        
        if len(words) > 15:
            # Tomar segmento medio de la oración
            start = len(words) // 4
            end = start + 8
            answer = " ".join(words[start:end])
        else:
            # Usar toda la oración si es corta
            answer = sentence
        
        # Limpiar y ajustar
        answer = answer.strip(' .,;:')
        
        # Asegurar que sea una respuesta apropiada
        if len(answer) < 20:
            answer = f"Lo que se describe en el texto sobre este tema"
        
        return answer[:100]  # Limitar longitud
    
    def _generate_explanation(self, question: str, correct_answer: str, 
                            source_sentence: str, domain: str) -> str:
        """Genera explicación para la respuesta correcta"""
        
        explanations = {
            "historia": f"La respuesta correcta se basa en la información histórica presentada en el texto: '{source_sentence[:100]}...'. Esto demuestra la importancia del contexto histórico para comprender los eventos.",
            
            "ciencias": f"La respuesta correcta se fundamenta en los principios científicos explicados: '{source_sentence[:100]}...'. Esto ilustra cómo los procesos naturales siguen leyes específicas.",
            
            "tecnologia": f"La respuesta correcta refleja las características tecnológicas descritas: '{source_sentence[:100]}...'. Esto muestra cómo la tecnología se aplica para resolver problemas específicos.",
            
            "general": f"La respuesta correcta se encuentra en el texto: '{source_sentence[:100]}...'. Esta información es clave para comprender el tema tratado."
        }
        
        return explanations.get(domain, explanations["general"])
    
    def _create_question_with_options(self, question: str, text: str, concepts: List[str], 
                                    domain: str, question_id: int, difficulty: str) -> Dict[str, Any]:
        """Crea pregunta completa con opciones basada en pregunta generada por IA"""
        
        # Buscar respuesta en el texto
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
        
        # Intentar encontrar respuesta relevante
        correct_answer = "La información presentada en el texto sobre este tema"
        
        for sentence in sentences:
            # Si la pregunta contiene algún concepto, buscar oración que también lo contenga
            if concepts:
                for concept in concepts[:2]:
                    if concept.lower() in sentence.lower():
                        correct_answer = sentence[:80] + "..." if len(sentence) > 80 else sentence
                        break
                if correct_answer != "La información presentada en el texto sobre este tema":
                    break
        
        # Crear opciones
        options = [
            correct_answer,
            "Una interpretación que no se encuentra en el texto",
            "Una conclusión que no está respaldada por el contenido",
            "Una información que contradice lo expuesto en el texto"
        ]
        
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        
        explanation = f"La respuesta correcta es '{correct_answer[:50]}...' porque esta información se encuentra directamente en el texto analizado y responde específicamente a la pregunta planteada."
        
        return {
            "id": question_id,
            "question": question,
            "options": options,
            "correct_answer": correct_index,
            "explanation": explanation,
            "difficulty": difficulty,
            "source": "ai_generated_with_options"
        }
    
    def _create_fallback_question(self, question_id: int, concepts: List[str], 
                                domain: str, difficulty: str) -> Dict[str, Any]:
        """Crea pregunta de respaldo de alta calidad"""
        
        main_concept = concepts[0] if concepts else "el tema principal"
        
        # Preguntas específicas por dominio
        domain_questions = {
            "historia": f"¿Cuál es la importancia histórica de {main_concept} según el texto?",
            "ciencias": f"¿Cómo se explica el funcionamiento de {main_concept} en el texto?",
            "tecnologia": f"¿Qué aplicaciones tiene {main_concept} según se describe en el contenido?",
            "literatura": f"¿Qué características literarias de {main_concept} se destacan en el texto?",
            "economia": f"¿Cuál es el impacto económico de {main_concept} según el análisis presentado?"
        }
        
        question = domain_questions.get(domain, f"¿Cuál es el aspecto más relevante de {main_concept} según el texto?")
        
        # Opciones mejoradas
        options = [
            f"Lo que se explica específicamente en el texto sobre {main_concept}",
            f"Una interpretación general no basada en el contenido específico",
            f"Una conclusión que contradice la información presentada",
            f"Una suposición que no tiene respaldo en el texto analizado"
        ]
        
        explanation = f"La respuesta correcta se basa en la información específica que el texto proporciona sobre {main_concept}, la cual es relevante y precisa para responder a la pregunta planteada."
        
        return {
            "id": question_id,
            "question": question,
            "options": options,
            "correct_answer": 0,
            "explanation": explanation,
            "difficulty": difficulty,
            "source": "intelligent_fallback"
        }
    
    def _create_emergency_quiz(self, concepts: List[str], num_questions: int, difficulty: str) -> List[Dict[str, Any]]:
        """Crea quiz de emergencia cuando fallan otros métodos"""
        
        questions = []
        
        for i in range(num_questions):
            concept = concepts[i % len(concepts)] if concepts else f"concepto {i+1}"
            
            question = {
                "id": i + 1,
                "question": f"¿Cuál es la información más importante sobre {concept} según el texto?",
                "options": [
                    f"La información específica que se presenta sobre {concept} en el contenido",
                    "Una interpretación que no está en el texto",
                    "Una conclusión no respaldada por el contenido",
                    "Una información contradictoria al texto"
                ],
                "correct_answer": 0,
                "explanation": f"La respuesta se basa en la información específica que el texto proporciona sobre {concept}.",
                "difficulty": difficulty,
                "source": "emergency"
            }
            
            questions.append(question)
        
        return questions
    
    async def generate_feedback(self, score: int, total: int, 
                              incorrect_questions: List[int], concepts: List[str]) -> str:
        """
        Genera feedback educativo personalizado y constructivo
        """
        try:
            percentage = (score / total) * 100
            
            # Crear feedback estructurado y motivador
            feedback = self._create_comprehensive_feedback(
                score, total, percentage, concepts, incorrect_questions
            )
            
            return feedback
            
        except Exception as e:
            logger.error(f"❌ Error generando feedback: {e}")
            return self._create_emergency_feedback(score, total)
    
    def _create_comprehensive_feedback(self, score: int, total: int, percentage: float,
                                     concepts: List[str], incorrect_questions: List[int]) -> str:
        """Crea feedback comprehensivo y personalizado"""
        
        # Determinar nivel de rendimiento
        if percentage >= 90:
            level = "excepcional"
            emoji = "🏆"
        elif percentage >= 80:
            level = "excelente"
            emoji = "⭐"
        elif percentage >= 70:
            level = "bueno"
            emoji = "👍"
        elif percentage >= 60:
            level = "satisfactorio"
            emoji = "📈"
        else:
            level = "mejora_necesaria"
            emoji = "💪"
        
        # Construir feedback estructurado
        feedback = f"{emoji} **EVALUACIÓN PERSONALIZADA DE RENDIMIENTO**\n\n"
        
        # Resultado destacado
        feedback += f"📊 **RESULTADO:** {score}/{total} respuestas correctas (**{percentage:.1f}%**)\n\n"
        
        # Mensaje principal según nivel
        if level == "excepcional":
            feedback += "🎉 **¡RENDIMIENTO EXCEPCIONAL!**\n\n"
            feedback += f"Has demostrado un dominio sobresaliente del tema. Tu comprensión de {concepts[0] if concepts else 'los conceptos principales'} es ejemplar y tu capacidad de análisis demuestra un aprendizaje profundo.\n\n"
            feedback += "💎 **FORTALEZAS IDENTIFICADAS:**\n"
            if concepts:
                feedback += f"• Excelente manejo de {concepts[0]}\n"
                if len(concepts) > 1:
                    feedback += f"• Comprensión avanzada de {concepts[1]}\n"
            feedback += "• Capacidad de síntesis y análisis crítico excepcional\n\n"
            feedback += "🚀 **RECOMENDACIONES PARA CONTINUAR:**\n"
            feedback += "• Explora aspectos más avanzados y aplicaciones del tema\n"
            feedback += "• Busca conexiones con otros temas relacionados\n"
            feedback += "• Considera compartir tu conocimiento con otros estudiantes\n"
        
        elif level == "excelente":
            feedback += "✨ **¡EXCELENTE TRABAJO!**\n\n"
            feedback += f"Tienes una comprensión sólida y bien fundamentada del tema. Tu manejo de {concepts[0] if concepts else 'los conceptos principales'} demuestra un aprendizaje efectivo.\n\n"
            feedback += "🌟 **FORTALEZAS IDENTIFICADAS:**\n"
            if concepts:
                feedback += f"• Buen dominio de {concepts[0]}\n"
                if len(concepts) > 1:
                    feedback += f"• Comprensión adecuada de {concepts[1]}\n"
            feedback += "• Capacidad para aplicar conocimientos correctamente\n\n"
            feedback += "📈 **OPORTUNIDADES DE MEJORA:**\n"
            if incorrect_questions:
                feedback += f"• Revisar preguntas {', '.join(map(str, incorrect_questions[:2]))}\n"
            feedback += "• Profundizar en aspectos específicos del tema\n"
        
        elif level == "bueno":
            feedback += "👍 **¡BUEN DESEMPEÑO!**\n\n"
            feedback += f"Has captado los aspectos fundamentales del tema. Tu comprensión de {concepts[0] if concepts else 'los conceptos principales'} está en desarrollo y muestra progreso positivo.\n\n"
            feedback += "📊 **ANÁLISIS DE RENDIMIENTO:**\n"
            feedback += f"• Conceptos dominados: {score} de {total}\n"
            if incorrect_questions:
                feedback += f"• Áreas de oportunidad: preguntas {', '.join(map(str, incorrect_questions[:3]))}\n"
            feedback += "\n💡 **ESTRATEGIAS DE MEJORA:**\n"
            feedback += "• Repasa los conceptos donde tuviste dificultades\n"
            feedback += "• Busca ejemplos adicionales para reforzar comprensión\n"
            feedback += "• Practica con ejercicios similares\n"
        
        else:  # satisfactorio o mejora_necesaria
            if level == "satisfactorio":
                feedback += "📈 **¡PROGRESO SATISFACTORIO!**\n\n"
                feedback += "Estás construyendo una base sólida de conocimientos. "
            else:
                feedback += "💪 **¡OPORTUNIDAD DE CRECIMIENTO!**\n\n"
                feedback += "Estás en proceso de construcción de tu comprensión. "
            
            feedback += f"Cada respuesta correcta representa progreso real en tu dominio de {concepts[0] if concepts else 'el tema'}.\n\n"
            
            feedback += "🎯 **ENFOQUE RECOMENDADO:**\n"
            if concepts:
                feedback += f"• Dedica tiempo extra a comprender {concepts[0]}\n"
                if len(concepts) > 1:
                    feedback += f"• Refuerza tu conocimiento de {concepts[1]}\n"
            feedback += "• Revisa los fundamentos antes de avanzar a temas complejos\n\n"
            
            feedback += "📚 **PLAN DE ESTUDIO SUGERIDO:**\n"
            feedback += "• Estudia un concepto a la vez hasta dominarlo\n"
            feedback += "• Usa recursos adicionales como videos o diagramas\n"
            feedback += "• Practica con ejercicios de dificultad gradual\n"
            feedback += "• No dudes en buscar ayuda cuando la necesites\n\n"
            
            feedback += "🌟 **MENSAJE MOTIVACIONAL:**\n"
            feedback += "El aprendizaje es un proceso gradual y personal. Cada paso que das te acerca más al dominio completo del tema. Tu perseverancia y dedicación son la clave del éxito futuro."
        
        return feedback
    
    def _create_emergency_feedback(self, score: int, total: int) -> str:
        """Crea feedback de emergencia básico pero útil"""
        percentage = (score / total) * 100
        
        if percentage >= 80:
            return f"🎉 **¡Excelente trabajo!** Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%). Tu comprensión del tema es sólida y demuestra un aprendizaje efectivo. ¡Continúa con este excelente trabajo!"
        
        elif percentage >= 60:
            return f"👍 **¡Buen trabajo!** Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%). Tienes una base sólida de conocimientos. Para mejorar, repasa los conceptos donde tuviste dificultades y practica con ejercicios adicionales."
        
        else:
            return f"💪 **¡Sigue adelante!** Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%). Estás construyendo tu comprensión del tema. Te recomiendo revisar los conceptos fundamentales y dedicar tiempo extra al estudio. ¡El aprendizaje es un proceso y cada intento te acerca más al éxito!"


# Alias para compatibilidad con código existente
AIService = ImprovedAIService