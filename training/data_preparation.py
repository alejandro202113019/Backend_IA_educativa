# training/data_preparation.py - GENERADOR DE DATOS UNIVERSAL MEJORADO
import random
import re
import logging
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
from collections import Counter
import json
import os

logger = logging.getLogger(__name__)

class UniversalEducationalDataGenerator:
    """
    Generador de datos sintéticos educativos para cualquier tipo de contenido
    """
    
    def __init__(self):
        logger.info("Inicializando generador universal de datos educativos")
        self._setup_universal_templates()
        self._load_base_datasets()
    
    def _setup_universal_templates(self):
        """Configura plantillas universales para diferentes tipos de contenido"""
        self.content_templates = {
            "historia": {
                "texts": [
                    "La Revolución Francesa (1789-1799) fue un período de cambios políticos y sociales radicales en Francia. Comenzó con la crisis financiera del Antiguo Régimen y culminó con el ascenso de Napoleón Bonaparte. Los principales eventos incluyeron la toma de la Bastilla, la Declaración de los Derechos del Hombre y la ejecución del rey Luis XVI.",
                    "La Segunda Guerra Mundial se desarrolló entre 1939 y 1945, involucrando a la mayoría de las naciones del mundo. El conflicto comenzó con la invasión alemana de Polonia y terminó con la rendición de Japón tras las bombas atómicas. Las principales potencias fueron los Aliados contra las Potencias del Eje.",
                    "El Imperio Romano alcanzó su máxima expansión durante el siglo II d.C. bajo el emperador Trajano. Se extendía desde Britania hasta Mesopotamia, abarcando todo el Mediterráneo. La Pax Romana permitió el desarrollo comercial y cultural durante más de dos siglos.",
                    "La Independencia de América se gestó durante el siglo XVIII como resultado de las tensiones entre las colonias y las metrópolis europeas. El proceso se caracterizó por la influencia de la Ilustración, las ideas liberales y la búsqueda de autonomía política y económica."
                ],
                "question_patterns": [
                    "¿Cuáles fueron las principales causas de {evento}?",
                    "¿Qué consecuencias tuvo {evento} en la sociedad de la época?",
                    "¿Cómo influyó {personaje} en el desarrollo de {evento}?",
                    "¿En qué período temporal se desarrolló {evento}?",
                    "¿Qué diferencias existen entre {evento1} y {evento2}?"
                ]
            },
            "ciencia": {
                "texts": [
                    "La fotosíntesis es el proceso bioquímico mediante el cual las plantas verdes y otros organismos convierten la energía lumínica en energía química. Este proceso ocurre en los cloroplastos y requiere dióxido de carbono, agua y luz solar para producir glucosa y oxígeno.",
                    "El ADN (ácido desoxirribonucleico) es una molécula que contiene la información genética de todos los seres vivos. Está formado por cuatro bases nitrogenadas: adenina, timina, guanina y citosina, que se combinan en secuencias específicas para formar genes.",
                    "La teoría de la evolución por selección natural, propuesta por Charles Darwin, explica cómo las especies cambian a lo largo del tiempo. Los organismos mejor adaptados a su ambiente tienen mayor probabilidad de sobrevivir y reproducirse, transmitiendo sus características a la descendencia.",
                    "El ciclo del agua es un proceso continuo de circulación del agua en la Tierra. Incluye la evaporación de océanos y ríos, la formación de nubes, la precipitación y el retorno del agua a los cuerpos de agua mediante escorrentía y infiltración."
                ],
                "question_patterns": [
                    "¿Cuál es la función principal de {proceso} en los organismos?",
                    "¿Qué elementos son necesarios para que ocurra {proceso}?",
                    "¿Cómo se relaciona {concepto1} con {concepto2}?",
                    "¿Qué importancia tiene {proceso} para la vida en la Tierra?",
                    "¿Cuáles son las etapas principales de {proceso}?"
                ]
            },
            "tecnologia": {
                "texts": [
                    "La inteligencia artificial es una rama de la informática que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. Incluye subcampos como machine learning, procesamiento de lenguaje natural y visión computacional.",
                    "La computación en la nube permite acceder a recursos informáticos (servidores, almacenamiento, software) a través de internet. Los principales modelos son Software as a Service (SaaS), Platform as a Service (PaaS) e Infrastructure as a Service (IaaS).",
                    "La ciberseguridad protege sistemas, redes y datos de ataques digitales. Incluye medidas como firewalls, cifrado, autenticación de dos factores y monitoreo continuo para prevenir accesos no autorizados y robo de información.",
                    "El desarrollo de software ágil es una metodología que enfatiza la colaboración, la adaptabilidad y la entrega incremental. Los principios ágiles incluyen individuos sobre procesos, software funcionando sobre documentación exhaustiva, y respuesta al cambio sobre seguir un plan."
                ],
                "question_patterns": [
                    "¿Cuáles son las principales aplicaciones de {tecnologia}?",
                    "¿Qué ventajas ofrece {tecnologia} sobre métodos tradicionales?",
                    "¿Cómo funciona {tecnologia} en términos básicos?",
                    "¿Qué desafíos presenta la implementación de {tecnologia}?",
                    "¿Cuál es el impacto de {tecnologia} en la sociedad actual?"
                ]
            },
            "literatura": {
                "texts": [
                    "El Realismo literario del siglo XIX se caracterizó por la representación objetiva de la realidad social. Autores como Gustave Flaubert y Honoré de Balzac retrataron la sociedad burguesa con precisión documental, abordando temas como la clase social y los conflictos económicos.",
                    "El Modernismo hispanoamericano fue un movimiento literario que renovó la poesía en español a finales del siglo XIX. Rubén Darío, su principal exponente, introdujo nuevas métricas, simbolismo y una estética refinada que influyó en toda la literatura en español.",
                    "El teatro del Siglo de Oro español alcanzó su apogeo con autores como Lope de Vega y Calderón de la Barca. Se caracterizó por la mezcla de lo trágico y lo cómico, la variedad métrica y el tratamiento de temas del honor y la religión.",
                    "La narrativa contemporánea latinoamericana experimentó con técnicas como el realismo mágico. Autores como Gabriel García Márquez y Jorge Luis Borges exploraron la realidad desde perspectivas innovadoras, mezclando elementos fantásticos con situaciones cotidianas."
                ],
                "question_patterns": [
                    "¿Cuáles son las características principales del movimiento {movimiento}?",
                    "¿Cómo influyó {autor} en el desarrollo de {genero}?",
                    "¿Qué temas recurrentes aparecen en la literatura de {epoca}?",
                    "¿Qué diferencias existen entre {movimiento1} y {movimiento2}?",
                    "¿Cuál fue el contexto histórico que influyó en {movimiento}?"
                ]
            },
            "economia": {
                "texts": [
                    "La oferta y la demanda son las fuerzas básicas que determinan los precios en una economía de mercado. La oferta representa la cantidad de un bien que los productores están dispuestos a vender a diferentes precios, mientras que la demanda representa la cantidad que los consumidores desean comprar.",
                    "La inflación es el aumento generalizado y sostenido de los precios de bienes y servicios en una economía. Sus principales causas incluyen el exceso de demanda, el aumento de los costos de producción y las expectativas inflacionarias de los agentes económicos.",
                    "El Producto Interno Bruto (PIB) mide el valor total de todos los bienes y servicios finales producidos en un país durante un período determinado. Es un indicador clave para evaluar el desempeño económico y comparar la productividad entre diferentes países.",
                    "Los mercados financieros facilitan el intercambio de activos financieros como acciones, bonos y divisas. Permiten que las empresas obtengan financiamiento para sus proyectos y que los inversionistas diversifiquen sus portafolios para maximizar rendimientos y minimizar riesgos."
                ],
                "question_patterns": [
                    "¿Cómo afecta {factor} al comportamiento de {mercado}?",
                    "¿Cuáles son las principales consecuencias de {fenomeno_economico}?",
                    "¿Qué relación existe entre {variable1} y {variable2}?",
                    "¿Cómo se calcula {indicador_economico}?",
                    "¿Qué políticas pueden implementarse para controlar {problema_economico}?"
                ]
            }
        }
        
        # Plantillas universales de feedback
        self.feedback_templates = {
            "excellent": [
                "¡Excelente trabajo! Has demostrado un dominio sobresaliente de {tema}. Tu comprensión de {concepto1} y {concepto2} es particularmente sólida.",
                "¡Felicitaciones! Tu rendimiento indica una comprensión profunda del material. Continúa explorando aspectos avanzados de {tema}.",
                "¡Impresionante! Has logrado conectar efectivamente los conceptos principales. Tu análisis de {concepto1} muestra madurez intelectual."
            ],
            "good": [
                "Buen trabajo. Has captado los conceptos principales de {tema}. Para mejorar, enfócate en profundizar tu comprensión de {concepto1}.",
                "Tu rendimiento es sólido. Tienes una buena base en {tema}, pero podrías beneficiarte de revisar {concepto1} y {concepto2}.",
                "Estás en el camino correcto. Tu comprensión de {tema} es buena, aunque hay algunas áreas como {concepto1} que necesitan refuerzo."
            ],
            "needs_improvement": [
                "Estás construyendo tu comprensión de {tema}. Te recomiendo revisar los conceptos fundamentales como {concepto1} y {concepto2}.",
                "Tu esfuerzo es notable. Para mejorar en {tema}, dedica tiempo extra a practicar con {concepto1} y busca ejemplos adicionales.",
                "Cada intento es progreso. Enfócate en dominar los fundamentos de {tema}, especialmente {concepto1}, antes de avanzar."
            ]
        }
    
    def _load_base_datasets(self):
        """Carga datasets públicos como base opcional"""
        try:
            logger.info("Intentando cargar datasets públicos...")
            # Cargar datasets solo si están disponibles
            try:
                self.cnn_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:500]")
                logger.info("Dataset CNN/DailyMail cargado")
            except:
                self.cnn_dataset = None
                logger.info("Dataset CNN/DailyMail no disponible")
            
            try:
                self.squad_dataset = load_dataset("squad_v2", split="train[:500]")
                logger.info("Dataset SQuAD cargado")
            except:
                self.squad_dataset = None
                logger.info("Dataset SQuAD no disponible")
                
        except Exception as e:
            logger.warning(f"Error cargando datasets públicos: {e}")
            self.cnn_dataset = None
            self.squad_dataset = None
    
    def generate_comprehensive_summary_data(self, n_samples: int = 800) -> List[Dict[str, Any]]:
        """Genera datos comprehensivos para fine-tuning de resúmenes universales"""
        logger.info(f"Generando {n_samples} ejemplos de resúmenes universales")
        
        summary_data = []
        
        # Distribución balanceada por tipo de contenido
        types = list(self.content_templates.keys())
        samples_per_type = n_samples // len(types)
        
        for content_type in types:
            logger.info(f"Generando {samples_per_type} ejemplos para {content_type}")
            
            for i in range(samples_per_type):
                try:
                    # Crear ejemplo sintético especializado
                    example = self._create_specialized_summary_example(content_type)
                    summary_data.append(example)
                    
                    if (i + 1) % 50 == 0:
                        logger.info(f"  Completados {i + 1}/{samples_per_type} para {content_type}")
                        
                except Exception as e:
                    logger.warning(f"Error generando ejemplo {i} para {content_type}: {e}")
                    continue
        
        # Usar datos de CNN/DailyMail si están disponibles
        if self.cnn_dataset and len(summary_data) < n_samples:
            remaining = n_samples - len(summary_data)
            cnn_examples = self._process_cnn_data(min(remaining, 200))
            summary_data.extend(cnn_examples)
        
        # Completar con ejemplos sintéticos adicionales si es necesario
        while len(summary_data) < n_samples:
            content_type = random.choice(types)
            example = self._create_specialized_summary_example(content_type)
            summary_data.append(example)
        
        logger.info(f"Generados {len(summary_data)} ejemplos de resúmenes universales")
        return summary_data[:n_samples]
    
    def _create_specialized_summary_example(self, content_type: str) -> Dict[str, Any]:
        """Crea ejemplo de resumen especializado por tipo de contenido"""
        
        templates = self.content_templates[content_type]
        base_text = random.choice(templates["texts"])
        
        # Expandir el texto base con variaciones
        expanded_text = self._expand_base_text(base_text, content_type)
        
        # Crear resumen educativo estructurado
        educational_summary = self._create_structured_educational_summary(expanded_text, content_type)
        
        # Extraer conceptos clave
        key_concepts = self._extract_concepts_from_text(expanded_text, content_type)
        
        return {
            "input_text": expanded_text,
            "target_text": educational_summary,
            "content_type": content_type,
            "concepts": key_concepts,
            "educational_level": random.choice(["básico", "intermedio", "avanzado"]),
            "length_category": "medium"
        }
    
    def _expand_base_text(self, base_text: str, content_type: str) -> str:
        """Expande el texto base con información adicional relevante"""
        
        expansions = {
            "historia": [
                " Este evento marcó un punto de inflexión en la historia occidental.",
                " Las repercusiones se sintieron durante décadas posteriores.",
                " Los historiadores consideran este período crucial para entender la época.",
                " Las fuentes primarias revelan la complejidad de los acontecimientos.",
                " El contexto político y social influyó significativamente en los resultados."
            ],
            "ciencia": [
                " Los científicos han estudiado este fenómeno durante décadas.",
                " Las investigaciones recientes han revelado nuevos aspectos importantes.",
                " Este proceso es fundamental para comprender sistemas más complejos.",
                " Las aplicaciones prácticas se extienden a múltiples disciplinas.",
                " Los experimentos controlados han confirmado las teorías propuestas."
            ],
            "tecnologia": [
                " Esta tecnología ha revolucionado la forma en que trabajamos.",
                " Las implementaciones exitosas demuestran su potencial transformador.",
                " Los desarrolladores continúan mejorando las funcionalidades existentes.",
                " La adopción masiva ha generado nuevas oportunidades de negocio.",
                " Los desafíos técnicos siguen siendo objeto de investigación activa."
            ],
            "literatura": [
                " Los críticos literarios han analizado extensamente estas obras.",
                " La influencia en autores posteriores es innegable.",
                " El contexto histórico-cultural enriqueció la producción literaria.",
                " Las técnicas narrativas empleadas fueron innovadoras para su época.",
                " El público contemporáneo recibió estas obras con gran interés."
            ],
            "economia": [
                " Los economistas han desarrollado modelos para explicar este fenómeno.",
                " Las políticas gubernamentales pueden influir significativamente en estos procesos.",
                " Los datos históricos confirman la importancia de estos factores.",
                " Las fluctuaciones del mercado reflejan estos principios fundamentales.",
                " Los análisis empíricos respaldan las predicciones teóricas."
            ]
        }
        
        # Agregar 1-2 expansiones aleatorias
        type_expansions = expansions.get(content_type, expansions["ciencia"])
        selected_expansions = random.sample(type_expansions, random.randint(1, 2))
        
        return base_text + " " + " ".join(selected_expansions)
    
    def _create_structured_educational_summary(self, text: str, content_type: str) -> str:
        """Crea resumen educativo estructurado específico por tipo"""
        
        # Extraer información clave
        key_info = self._extract_key_information(text, content_type)
        
        # Estructura base
        summary = "📚 **RESUMEN EDUCATIVO**\n\n"
        
        # Tipo específico de encabezado
        type_headers = {
            "historia": "🏛️ **ANÁLISIS HISTÓRICO**",
            "ciencia": "🔬 **ANÁLISIS CIENTÍFICO**", 
            "tecnologia": "💻 **ANÁLISIS TECNOLÓGICO**",
            "literatura": "📖 **ANÁLISIS LITERARIO**",
            "economia": "📊 **ANÁLISIS ECONÓMICO**"
        }
        
        summary += f"{type_headers.get(content_type, '🎯 **ANÁLISIS TEMÁTICO**')}\n\n"
        
        # Conceptos clave
        if key_info.get("concepts"):
            summary += f"🔑 **CONCEPTOS CLAVE:** {', '.join(key_info['concepts'][:4])}\n\n"
        
        # Información específica por tipo
        if content_type == "historia" and key_info.get("period"):
            summary += f"📅 **PERÍODO:** {key_info['period']}\n\n"
        
        if content_type == "ciencia" and key_info.get("process"):
            summary += f"⚗️ **PROCESO PRINCIPAL:** {key_info['process']}\n\n"
        
        # Contenido principal (resumen del texto)
        main_content = self._generate_main_content_summary(text, content_type)
        summary += f"📝 **CONTENIDO PRINCIPAL:**\n{main_content}\n\n"
        
        # Puntos importantes
        key_points = self._generate_content_specific_points(text, content_type)
        if key_points:
            summary += f"💡 **PUNTOS IMPORTANTES:**\n"
            for i, point in enumerate(key_points, 1):
                summary += f"{i}. {point}\n"
        
        return summary
    
    def _extract_key_information(self, text: str, content_type: str) -> Dict[str, Any]:
        """Extrae información clave específica por tipo de contenido"""
        
        info = {"concepts": [], "period": None, "process": None}
        
        if content_type == "historia":
            # Extraer períodos y fechas
            periods = re.findall(r'\b(?:siglo\s+[IVX]+|(?:17|18|19|20)\d{2})', text, re.IGNORECASE)
            if periods:
                info["period"] = periods[0]
            
            # Conceptos históricos
            historical_terms = re.findall(r'\b(?:revolución|guerra|imperio|independencia|reforma|tratado)\w*\b', text, re.IGNORECASE)
            info["concepts"].extend(list(set(historical_terms))[:3])
        
        elif content_type == "ciencia":
            # Procesos científicos
            processes = re.findall(r'\b(?:fotosíntesis|evolución|ciclo|proceso|reacción)\w*\b', text, re.IGNORECASE)
            if processes:
                info["process"] = processes[0]
            
            # Términos científicos
            science_terms = re.findall(r'\b(?:célula|gen|proteína|energía|molécula|organismo)\w*\b', text, re.IGNORECASE)
            info["concepts"].extend(list(set(science_terms))[:3])
        
        elif content_type == "tecnologia":
            # Términos tecnológicos
            tech_terms = re.findall(r'\b(?:algoritmo|software|sistema|inteligencia|datos|red)\w*\b', text, re.IGNORECASE)
            info["concepts"].extend(list(set(tech_terms))[:3])
        
        # Conceptos generales (palabras importantes)
        general_concepts = re.findall(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{4,}\b', text)
        concept_freq = Counter(general_concepts)
        
        # Filtrar stop words
        stop_words = {'Este', 'Esta', 'Estos', 'Estas', 'Para', 'Según', 'Durante', 'Después'}
        filtered_concepts = [word for word, freq in concept_freq.most_common(5) 
                           if word not in stop_words and freq >= 1]
        
        info["concepts"].extend(filtered_concepts[:3])
        info["concepts"] = list(set(info["concepts"]))[:5]  # Eliminar duplicados y limitar
        
        return info
    
    def _generate_main_content_summary(self, text: str, content_type: str) -> str:
        """Genera resumen del contenido principal"""
        
        # Extraer las 2-3 oraciones más importantes
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
        
        if len(sentences) <= 2:
            return " ".join(sentences) + "."
        
        # Seleccionar oraciones representativas
        if len(sentences) >= 3:
            # Primera, una del medio, y última
            selected = [sentences[0], sentences[len(sentences)//2], sentences[-1]]
        else:
            selected = sentences
        
        summary_text = " ".join(selected)
        
        # Añadir conectores según el tipo
        if content_type == "historia":
            summary_text = summary_text.replace(". ", ". Posteriormente, ")
        elif content_type == "ciencia":
            summary_text = summary_text.replace(". ", ". En este proceso, ")
        
        return summary_text + "."
    
    def _generate_content_specific_points(self, text: str, content_type: str) -> List[str]:
        """Genera puntos específicos según el tipo de contenido"""
        
        points = []
        
        if content_type == "historia":
            points = [
                "Los eventos históricos están interconectados con factores políticos, sociales y económicos",
                "El contexto temporal es crucial para comprender las motivaciones de los actores históricos",
                "Las consecuencias de estos eventos influenciaron el desarrollo posterior de la sociedad"
            ]
        
        elif content_type == "ciencia":
            points = [
                "Los procesos científicos siguen leyes naturales que pueden ser estudiadas y comprendidas",
                "La experimentación y observación son fundamentales para validar las teorías científicas",
                "Los avances científicos tienen aplicaciones prácticas que benefician a la sociedad"
            ]
        
        elif content_type == "tecnologia":
            points = [
                "Las tecnologías evolucionan constantemente para satisfacer necesidades humanas",
                "La implementación exitosa requiere consideración de factores técnicos y sociales",
                "El impacto tecnológico se extiende más allá de su aplicación inmediata"
            ]
        
        elif content_type == "literatura":
            points = [
                "Las obras literarias reflejan el contexto histórico y cultural de su época",
                "Los recursos estilísticos empleados enriquecen la expresión artística",
                "La literatura influye en la formación de identidades culturales y valores sociales"
            ]
        
        elif content_type == "economia":
            points = [
                "Los fenómenos económicos están influenciados por múltiples variables interrelacionadas",
                "Las políticas económicas tienen efectos tanto a corto como a largo plazo",
                "La comprensión económica es esencial para la toma de decisiones informadas"
            ]
        
        return points[:3]
    
    def generate_comprehensive_question_data(self, n_samples: int = 600) -> List[Dict[str, Any]]:
        """Genera datos comprehensivos para fine-tuning de generación de preguntas"""
        logger.info(f"Generando {n_samples} ejemplos de preguntas universales")
        
        question_data = []
        
        # Distribución balanceada por tipo de contenido
        types = list(self.content_templates.keys())
        samples_per_type = n_samples // len(types)
        
        for content_type in types:
            logger.info(f"Generando {samples_per_type} preguntas para {content_type}")
            
            for i in range(samples_per_type):
                try:
                    example = self._create_specialized_question_example(content_type)
                    question_data.append(example)
                    
                    if (i + 1) % 25 == 0:
                        logger.info(f"  Completadas {i + 1}/{samples_per_type} para {content_type}")
                        
                except Exception as e:
                    logger.warning(f"Error generando pregunta {i} para {content_type}: {e}")
                    continue
        
        # Usar datos de SQuAD si están disponibles
        if self.squad_dataset and len(question_data) < n_samples:
            remaining = n_samples - len(question_data)
            squad_examples = self._process_squad_data(min(remaining, 100))
            question_data.extend(squad_examples)
        
        # Completar con ejemplos sintéticos si es necesario
        while len(question_data) < n_samples:
            content_type = random.choice(types)
            example = self._create_specialized_question_example(content_type)
            question_data.append(example)
        
        logger.info(f"Generadas {len(question_data)} preguntas universales")
        return question_data[:n_samples]
    
    def _create_specialized_question_example(self, content_type: str) -> Dict[str, Any]:
        """Crea ejemplo de pregunta especializada por tipo de contenido"""
        
        templates = self.content_templates[content_type]
        base_text = random.choice(templates["texts"])
        question_patterns = templates["question_patterns"]
        
        # Expandir texto
        expanded_text = self._expand_base_text(base_text, content_type)
        
        # Extraer conceptos para la pregunta
        concepts = self._extract_concepts_from_text(expanded_text, content_type)
        
        # Generar pregunta educativa
        question = self._generate_educational_question(expanded_text, concepts, content_type, question_patterns)
        
        # Crear input prompt
        input_prompt = f"""Contexto sobre {content_type}: {expanded_text}

Genera una pregunta educativa de calidad sobre este contenido:"""
        
        return {
            "input_text": input_prompt,
            "target_text": question,
            "content_type": content_type,
            "context": expanded_text,
            "concepts": concepts,
            "difficulty": random.choice(["fácil", "medio", "difícil"])
        }
    
    def _generate_educational_question(self, text: str, concepts: List[str], 
                                     content_type: str, patterns: List[str]) -> str:
        """Genera pregunta educativa de calidad"""
        
        # Seleccionar patrón aleatorio
        pattern = random.choice(patterns)
        
        # Llenar placeholders con conceptos extraídos
        if concepts:
            if "{evento}" in pattern:
                pattern = pattern.replace("{evento}", concepts[0])
            if "{proceso}" in pattern:
                pattern = pattern.replace("{proceso}", concepts[0])
            if "{concepto}" in pattern or "{concepto1}" in pattern:
                pattern = pattern.replace("{concepto}", concepts[0])
                pattern = pattern.replace("{concepto1}", concepts[0])
            if "{concepto2}" in pattern and len(concepts) > 1:
                pattern = pattern.replace("{concepto2}", concepts[1])
            if "{personaje}" in pattern:
                # Buscar nombres propios en el texto
                names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)
                if names:
                    pattern = pattern.replace("{personaje}", names[0])
                else:
                    pattern = pattern.replace("{personaje}", "los personajes principales")
        
        # Limpiar placeholders no reemplazados
        pattern = re.sub(r'\{[^}]+\}', 'los elementos principales', pattern)
        
        return pattern
    
    def generate_comprehensive_feedback_data(self, n_samples: int = 400) -> List[Dict[str, Any]]:
        """Genera datos comprehensivos para fine-tuning de feedback"""
        logger.info(f"Generando {n_samples} ejemplos de feedback universal")
        
        feedback_data = []
        
        # Distribución por tipos de rendimiento
        performance_levels = ["excellent", "good", "needs_improvement"]
        samples_per_level = n_samples // len(performance_levels)
        
        for level in performance_levels:
            logger.info(f"Generando {samples_per_level} ejemplos de feedback {level}")
            
            for i in range(samples_per_level):
                try:
                    example = self._create_feedback_example(level)
                    feedback_data.append(example)
                    
                    if (i + 1) % 25 == 0:
                        logger.info(f"  Completados {i + 1}/{samples_per_level} para {level}")
                        
                except Exception as e:
                    logger.warning(f"Error generando feedback {i} para {level}: {e}")
                    continue
        
        # Completar con ejemplos adicionales
        while len(feedback_data) < n_samples:
            level = random.choice(performance_levels)
            example = self._create_feedback_example(level)
            feedback_data.append(example)
        
        logger.info(f"Generados {len(feedback_data)} ejemplos de feedback")
        return feedback_data[:n_samples]
    
    def _create_feedback_example(self, performance_level: str) -> Dict[str, Any]:
        """Crea ejemplo de feedback personalizado"""
        
        # Simular resultados de quiz según nivel
        if performance_level == "excellent":
            total_questions = random.randint(5, 12)
            score = random.randint(int(total_questions * 0.9), total_questions)
        elif performance_level == "good":
            total_questions = random.randint(5, 12)
            score = random.randint(int(total_questions * 0.7), int(total_questions * 0.89))
        else:  # needs_improvement
            total_questions = random.randint(5, 12)
            score = random.randint(0, int(total_questions * 0.69))
        
        percentage = (score / total_questions) * 100
        
        # Seleccionar tipo de contenido y conceptos
        content_type = random.choice(list(self.content_templates.keys()))
        concepts = self._get_sample_concepts(content_type)
        
        # Simular preguntas incorrectas
        incorrect_count = total_questions - score
        incorrect_questions = random.sample(range(1, total_questions + 1), incorrect_count)
        
        # Crear input
        input_text = f"""Resultados del quiz sobre {content_type}:
Puntuación: {score}/{total_questions} ({percentage:.1f}%)
Conceptos evaluados: {', '.join(concepts)}
Preguntas incorrectas: {incorrect_questions if incorrect_questions else 'Ninguna'}
Tipo de contenido: {content_type}

Genera feedback educativo personalizado y constructivo:"""
        
        # Generar feedback objetivo
        target_feedback = self._generate_structured_feedback(
            score, total_questions, percentage, concepts, content_type, performance_level
        )
        
        return {
            "input_text": input_text,
            "target_text": target_feedback,
            "performance_level": performance_level,
            "content_type": content_type,
            "score": score,
            "total": total_questions,
            "concepts": concepts
        }
    
    def _get_sample_concepts(self, content_type: str) -> List[str]:
        """Obtiene conceptos de muestra para un tipo de contenido"""
        
        concept_samples = {
            "historia": ["revolución", "imperio", "guerra", "independencia", "tratado"],
            "ciencia": ["fotosíntesis", "evolución", "célula", "energía", "ADN"],
            "tecnologia": ["algoritmo", "inteligencia artificial", "software", "datos", "sistema"],
            "literatura": ["modernismo", "realismo", "narrativa", "poesía", "estilo"],
            "economia": ["mercado", "inflación", "PIB", "oferta", "demanda"]
        }
        
        concepts = concept_samples.get(content_type, ["concepto1", "concepto2", "concepto3"])
        return random.sample(concepts, random.randint(2, 4))
    
    def _generate_structured_feedback(self, score: int, total: int, percentage: float,
                                    concepts: List[str], content_type: str, level: str) -> str:
        """Genera feedback estructurado y personalizado"""
        
        # Seleccionar plantilla base
        templates = self.feedback_templates[level]
        base_template = random.choice(templates)
        
        # Reemplazar placeholders
        tema = content_type.replace("_", " ").title()
        concepto1 = concepts[0] if concepts else "los conceptos principales"
        concepto2 = concepts[1] if len(concepts) > 1 else "los temas fundamentales"
        
        feedback = base_template.format(
            tema=tema,
            concepto1=concepto1,
            concepto2=concepto2
        )
        
        # Agregar información específica
        feedback += f"\n\n📊 **RESULTADO DETALLADO:** {score}/{total} respuestas correctas ({percentage:.1f}%)\n"
        
        # Recomendaciones específicas por tipo de contenido
        if content_type == "historia":
            if level == "excellent":
                feedback += "\n🎯 **RECOMENDACIÓN:** Explora fuentes primarias y diferentes perspectivas históricas."
            else:
                feedback += "\n🎯 **RECOMENDACIÓN:** Crea líneas de tiempo y estudia mapas históricos para mejor comprensión."
        
        elif content_type == "ciencia":
            if level == "excellent":
                feedback += "\n🎯 **RECOMENDACIÓN:** Investiga aplicaciones prácticas y experimentos relacionados."
            else:
                feedback += "\n🎯 **RECOMENDACIÓN:** Repasa principios fundamentales y practica con diagramas explicativos."
        
        elif content_type == "tecnologia":
            if level == "excellent":
                feedback += "\n🎯 **RECOMENDACIÓN:** Experimenta con herramientas y explora tendencias emergentes."
            else:
                feedback += "\n🎯 **RECOMENDACIÓN:** Familiarízate con terminología básica y practica con ejemplos step-by-step."
        
        # Mensaje motivacional
        if level == "needs_improvement":
            feedback += "\n\n🌟 **RECUERDA:** Cada intento es una oportunidad de aprendizaje. Tu perseverancia es clave para el éxito."
        
        return feedback
    
    def _extract_concepts_from_text(self, text: str, content_type: str) -> List[str]:
        """Extrae conceptos específicos según el tipo de contenido"""
        
        # Patrones específicos por tipo
        type_patterns = {
            "historia": r'\b(?:guerra|batalla|revolución|imperio|rey|emperador|tratado|independencia)\w*\b',
            "ciencia": r'\b(?:célula|gen|proteína|energía|proceso|teoría|evolución|especie)\w*\b',
            "tecnologia": r'\b(?:algoritmo|software|sistema|datos|inteligencia|artificial|programa)\w*\b',
            "literatura": r'\b(?:obra|autor|estilo|narrativa|poesía|movimiento|género)\w*\b',
            "economia": r'\b(?:mercado|precio|inflación|demanda|oferta|PIB|economía)\w*\b'
        }
        
        # Buscar conceptos específicos
        pattern = type_patterns.get(content_type, r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{4,}\b')
        specific_concepts = re.findall(pattern, text, re.IGNORECASE)
        
        # Buscar conceptos generales
        general_concepts = re.findall(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{4,}\b', text)
        
        # Combinar y limpiar
        all_concepts = list(set(specific_concepts + general_concepts))
        
        # Filtrar stop words
        stop_words = {'Este', 'Esta', 'Estos', 'Estas', 'Para', 'Todo', 'Cada', 'Debe', 'Puede'}
        filtered_concepts = [concept.title() for concept in all_concepts 
                           if concept.title() not in stop_words]
        
        # Contar frecuencias y retornar los más relevantes
        concept_freq = Counter([concept.lower() for concept in filtered_concepts])
        return [concept.title() for concept, _ in concept_freq.most_common(6)]
    
    def _process_cnn_data(self, n_samples: int) -> List[Dict[str, Any]]:
        """Procesa datos de CNN/DailyMail para resúmenes"""
        processed_data = []
        
        try:
            for i, item in enumerate(self.cnn_dataset):
                if len(processed_data) >= n_samples:
                    break
                
                # Convertir a formato educativo
                educational_summary = self._convert_to_educational_summary(
                    item['article'], item['highlights']
                )
                processed_data.append(educational_summary)
                
        except Exception as e:
            logger.warning(f"Error procesando datos CNN: {e}")
        
        return processed_data
    
    def _process_squad_data(self, n_samples: int) -> List[Dict[str, Any]]:
        """Procesa datos de SQuAD para preguntas"""
        processed_data = []
        
        try:
            for i, item in enumerate(self.squad_dataset):
                if len(processed_data) >= n_samples:
                    break
                
                # Convertir a formato educativo
                educational_qa = self._convert_to_educational_qa(item)
                processed_data.append(educational_qa)
                
        except Exception as e:
            logger.warning(f"Error procesando datos SQuAD: {e}")
        
        return processed_data
    
    def _convert_to_educational_summary(self, article: str, highlights: str) -> Dict[str, Any]:
        """Convierte datos de CNN a formato educativo"""
        
        # Detectar tipo de contenido aproximado
        content_type = "general"
        if any(word in article.lower() for word in ["government", "election", "president"]):
            content_type = "historia"
        elif any(word in article.lower() for word in ["study", "research", "scientists"]):
            content_type = "ciencia"
        elif any(word in article.lower() for word in ["technology", "software", "app"]):
            content_type = "tecnologia"
        
        # Crear resumen educativo
        educational_summary = f"""📚 **RESUMEN EDUCATIVO**

📝 **CONTENIDO PRINCIPAL:**
{highlights}

💡 **INFORMACIÓN ADICIONAL:**
Este contenido aborda aspectos importantes que requieren análisis cuidadoso y comprensión integral."""
        
        return {
            "input_text": article[:800],
            "target_text": educational_summary,
            "content_type": content_type,
            "concepts": self._extract_concepts_from_text(article, content_type),
            "educational_level": "intermedio"
        }
    
    def _convert_to_educational_qa(self, squad_item: Dict) -> Dict[str, Any]:
        """Convierte datos de SQuAD a formato educativo"""
        
        context = squad_item['context']
        question = squad_item['question']
        
        # Mejorar pregunta para ser más educativa
        educational_question = self._enhance_question_educational(question)
        
        # Crear prompt
        input_prompt = f"""Contexto: {context}

Genera una pregunta educativa sobre este contenido:"""
        
        return {
            "input_text": input_prompt,
            "target_text": educational_question,
            "content_type": "general",
            "context": context,
            "difficulty": "medio"
        }
    
    def _enhance_question_educational(self, question: str) -> str:
        """Mejora una pregunta para hacerla más educativa"""
        
        educational_starters = [
            "¿Cómo se puede explicar",
            "¿Cuál es la importancia de",
            "¿Qué factores contribuyen a",
            "¿Por qué es relevante",
            "¿De qué manera influye"
        ]
        
        # 30% de probabilidad de mejorar con starter educativo
        if random.random() < 0.3:
            starter = random.choice(educational_starters)
            return f"{starter} {question.lower()}?"
        
        return question
    
    def save_training_data(self, data: List[Dict], filename: str) -> None:
        """Guarda los datos de entrenamiento en formato JSON"""
        try:
            output_path = f"./training_data/{filename}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Datos guardados en {output_path}")
            
        except Exception as e:
            logger.error(f"Error guardando datos: {e}")
    
    def generate_complete_training_dataset(self, 
                                         summary_samples: int = 800,
                                         question_samples: int = 600, 
                                         feedback_samples: int = 400) -> Dict[str, List[Dict]]:
        """Genera dataset completo para entrenamiento universal"""
        
        logger.info("🚀 Generando dataset completo para entrenamiento universal")
        
        # Generar todos los tipos de datos
        summary_data = self.generate_comprehensive_summary_data(summary_samples)
        question_data = self.generate_comprehensive_question_data(question_samples)
        feedback_data = self.generate_comprehensive_feedback_data(feedback_samples)
        
        # Crear dataset completo
        complete_dataset = {
            "summary_data": summary_data,
            "question_data": question_data,
            "feedback_data": feedback_data,
            "metadata": {
                "total_samples": len(summary_data) + len(question_data) + len(feedback_data),
                "content_types": list(self.content_templates.keys()),
                "generation_date": "2024-01-01",
                "version": "universal_v1.0"
            }
        }
        
        logger.info(f"✅ Dataset completo generado:")
        logger.info(f"   📝 Resúmenes: {len(summary_data)} ejemplos")
        logger.info(f"   ❓ Preguntas: {len(question_data)} ejemplos")
        logger.info(f"   💬 Feedback: {len(feedback_data)} ejemplos")
        logger.info(f"   📊 Total: {complete_dataset['metadata']['total_samples']} ejemplos")
        
        return complete_dataset