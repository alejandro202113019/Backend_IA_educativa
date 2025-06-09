# app/services/improved_quiz_generator.py - GENERADOR INTELIGENTE DE QUIZ CONTEXTUAL
import re
import random
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import spacy
from datetime import datetime

logger = logging.getLogger(__name__)

class IntelligentQuizGenerator:
    """
    Generador inteligente de quiz que crea preguntas específicas del contenido analizado
    """
    
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self._setup_contextual_patterns()
        self._setup_question_templates()
        self._setup_domain_specific_patterns()
    
    def _load_spacy_model(self):
        """Carga modelo de spaCy para análisis avanzado"""
        models_to_try = ["es_core_news_lg", "es_core_news_md", "es_core_news_sm"]
        
        for model_name in models_to_try:
            try:
                nlp = spacy.load(model_name)
                logger.info(f"✅ Modelo spaCy cargado: {model_name}")
                return nlp
            except OSError:
                continue
        
        logger.warning("⚠️ No se pudo cargar modelo de spaCy, usando análisis básico")
        return None
    
    def _setup_contextual_patterns(self):
        """Configura patrones para extraer información contextual específica"""
        self.content_extractors = {
            "fechas_historicas": r'\b(?:19[0-9]{2}|20[0-2][0-9])\b',
            "periodos": r'\b(?:siglo\s+[IVX]+|(?:primera|segunda)\s+guerra\s+mundial)\b',
            "personajes_historicos": r'\b(?:Hitler|Stalin|Roosevelt|Churchill|Mussolini|Franco)\b',
            "lugares_geograficos": r'\b(?:Europa|Asia|África|América|Alemania|Francia|Italia|España|Polonia|URSS|Estados Unidos|Reino Unido)\b',
            "eventos_militares": r'\b(?:batalla|operación|invasión|bombardeo|ofensiva|desembarco|blitzkrieg)\b',
            "conceptos_politicos": r'\b(?:fascismo|nazismo|comunismo|democracia|totalitarismo|propaganda)\b',
            "organizaciones": r'\b(?:Wehrmacht|SS|Gestapo|Luftwaffe|Sociedad de Naciones|ONU)\b',
            "consecuencias": r'\b(?:consecuencias?|efectos?|resultados?|impacto)\b',
            "causas": r'\b(?:causas?|origen|motivos?|razones?)\b'
        }
        
        # Patrones para extraer datos específicos de la Segunda Guerra Mundial
        self.wwii_specific_patterns = {
            "fases_guerra": r'\b(?:1939-1941|1941-1943|1943-1945|fase inicial|punto de inflexión|contraofensiva)\b',
            "frentes": r'\b(?:frente occidental|frente oriental|teatro del pacífico|frente africano)\b',
            "tecnologia_militar": r'\b(?:radar|tanque|submarino|avión|bomba atómica|misil|V-2)\b',
            "ideologias": r'\b(?:Tercer Reich|Eje|Aliados|New Deal|Plan Marshall)\b',
            "tratados": r'\b(?:tratado|pacto|acuerdo|rendición|armisticio)\b'
        }
    
    def _setup_question_templates(self):
        """Configura plantillas inteligentes de preguntas por categoría"""
        self.question_templates = {
            "cronologia": [
                "¿En qué año {evento_especifico}?",
                "¿Cuándo comenzó {proceso_historico}?",
                "¿Qué período abarca {evento_contexto}?",
                "¿En qué fase de la guerra ocurrió {evento_militar}?",
                "¿Cuál fue la secuencia temporal de {eventos_relacionados}?"
            ],
            "personajes": [
                "¿Qué papel desempeñó {personaje} en {contexto_especifico}?",
                "¿Cuál fue la estrategia de {lider} durante {periodo}?",
                "¿Cómo influyó {personaje} en {evento_especifico}?",
                "¿Qué decisión tomó {lider} respecto a {situacion}?",
                "¿Por qué {personaje} fue importante para {resultado}?"
            ],
            "geografia": [
                "¿Dónde se desarrolló {evento_militar}?",
                "¿Qué países formaron {alianza_especifica}?",
                "¿Cuál fue la importancia estratégica de {lugar}?",
                "¿Qué territorios conquistó {pais} durante {periodo}?",
                "¿Por qué {lugar} fue clave en {operacion_militar}?"
            ],
            "causas_consecuencias": [
                "¿Cuáles fueron las principales causas de {evento}?",
                "¿Qué consecuencias tuvo {evento} para {region_grupo}?",
                "¿Por qué {evento} cambió {aspecto_especifico}?",
                "¿Cómo afectó {evento} al desarrollo de {consecuencia}?",
                "¿Qué factores llevaron a {resultado_especifico}?"
            ],
            "conceptos": [
                "¿Qué caracterizó al {concepto_politico} durante {periodo}?",
                "¿Cómo funcionaba {sistema_organizacion}?",
                "¿En qué consistía la estrategia de {tactica_militar}?",
                "¿Qué principios definían {ideologia}?",
                "¿Cuál era el objetivo de {politica_estrategia}?"
            ],
            "comparacion": [
                "¿Qué diferencias había entre {elemento1} y {elemento2}?",
                "¿Cómo se comparaba {aspecto} de {pais1} con {pais2}?",
                "¿En qué se diferenciaron las estrategias de {lider1} y {lider2}?",
                "¿Qué ventajas tenía {bando1} sobre {bando2}?",
                "¿Cómo varió {aspecto} entre {periodo1} y {periodo2}?"
            ]
        }
    
    def _setup_domain_specific_patterns(self):
        """Configura patrones específicos para diferentes dominios"""
        self.domain_patterns = {
            "segunda_guerra_mundial": {
                "eventos_clave": [
                    "invasión de Polonia", "batalla de Francia", "batalla de Inglaterra",
                    "operación Barbarroja", "ataque a Pearl Harbor", "batalla de Stalingrado",
                    "desembarco de Normandía", "bomba atómica", "rendición de Alemania"
                ],
                "personajes_principales": [
                    "Adolf Hitler", "Winston Churchill", "Franklin Roosevelt", 
                    "José Stalin", "Benito Mussolini", "Charles de Gaulle"
                ],
                "conceptos_estrategicos": [
                    "Blitzkrieg", "guerra total", "resistencia", "colaboracionismo",
                    "genocidio", "holocausto", "guerra fría"
                ]
            }
        }
    
    def generate_contextual_quiz(self, text: str, num_questions: int = 5, 
                               difficulty: str = "medium") -> List[Dict[str, Any]]:
        """
        Genera quiz contextual inteligente basado en el contenido específico del texto
        """
        logger.info(f"🎯 Generando quiz contextual de {num_questions} preguntas")
        
        # 1. Análizar contenido específico del texto
        content_analysis = self._analyze_text_content(text)
        
        # 2. Extraer información factual específica
        factual_data = self._extract_factual_information(text, content_analysis)
        
        # 3. Generar preguntas basadas en contenido real
        questions = []
        
        # Distribuir preguntas por categorías
        categories = ["cronologia", "personajes", "geografia", "causas_consecuencias", "conceptos"]
        questions_per_category = max(1, num_questions // len(categories))
        
        for category in categories:
            if len(questions) >= num_questions:
                break
                
            category_questions = self._generate_category_questions(
                category, factual_data, content_analysis, difficulty, questions_per_category
            )
            questions.extend(category_questions)
        
        # Completar con preguntas adicionales si es necesario
        while len(questions) < num_questions:
            additional_question = self._generate_direct_content_question(
                text, factual_data, len(questions) + 1, difficulty
            )
            if additional_question:
                questions.append(additional_question)
            else:
                break
        
        # Validar y mejorar calidad de preguntas
        validated_questions = self._validate_and_improve_questions(questions, text)
        
        logger.info(f"✅ Generadas {len(validated_questions)} preguntas contextuales")
        return validated_questions[:num_questions]
    
    def _analyze_text_content(self, text: str) -> Dict[str, Any]:
        """Analiza el contenido específico del texto"""
        analysis = {
            "domain": "historia",
            "specific_topic": "segunda_guerra_mundial",
            "key_entities": [],
            "temporal_info": [],
            "geographical_info": [],
            "key_concepts": [],
            "main_themes": []
        }
        
        # Detectar dominio específico
        if "segunda guerra mundial" in text.lower():
            analysis["specific_topic"] = "segunda_guerra_mundial"
        elif "primera guerra mundial" in text.lower():
            analysis["specific_topic"] = "primera_guerra_mundial"
        
        # Extraer entidades específicas usando patrones
        for pattern_name, pattern in self.content_extractors.items():
            matches = list(set(re.findall(pattern, text, re.IGNORECASE)))
            if matches:
                analysis[pattern_name] = matches
        
        # Usar spaCy si está disponible para análisis más profundo
        if self.nlp:
            doc = self.nlp(text)
            
            # Extraer entidades nombradas
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'GPE', 'ORG', 'EVENT']:
                    analysis["key_entities"].append({
                        "text": ent.text,
                        "label": ent.label_,
                        "context": text[max(0, ent.start_char-50):ent.end_char+50]
                    })
        
        return analysis
    
    def _extract_factual_information(self, text: str, analysis: Dict) -> Dict[str, Any]:
        """Extrae información factual específica del texto"""
        factual_data = {
            "dates": [],
            "people": [],
            "places": [],
            "events": [],
            "concepts": [],
            "relationships": [],
            "specific_facts": []
        }
        
        # Extraer fechas con contexto
        date_pattern = r'(\b(?:19[0-9]{2}|20[0-2][0-9])\b)'
        for match in re.finditer(date_pattern, text):
            date = match.group(1)
            context_start = max(0, match.start() - 100)
            context_end = min(len(text), match.end() + 100)
            context = text[context_start:context_end].strip()
            
            factual_data["dates"].append({
                "date": date,
                "context": context,
                "position": match.start()
            })
        
        # Extraer información específica de WWII si es el tema
        if analysis.get("specific_topic") == "segunda_guerra_mundial":
            factual_data.update(self._extract_wwii_specific_facts(text))
        
        # Extraer relaciones causa-efecto
        factual_data["relationships"] = self._extract_causal_relationships(text)
        
        # Extraer hechos específicos del texto (oraciones con información factual)
        factual_data["specific_facts"] = self._extract_specific_facts(text)
        
        return factual_data
    
    def _extract_wwii_specific_facts(self, text: str) -> Dict[str, Any]:
        """Extrae hechos específicos sobre la Segunda Guerra Mundial"""
        wwii_facts = {
            "war_phases": [],
            "military_operations": [],
            "key_battles": [],
            "alliances": [],
            "technologies": [],
            "casualties": [],
            "political_changes": []
        }
        
        # Buscar información sobre fases de la guerra
        phase_patterns = {
            "inicio": r'(1939.*?(?:invasión|Polonia|comenzó|inició))',
            "expansion": r'(1940.*?(?:Francia|Inglaterra|batalla))',
            "globalizacion": r'(1941.*?(?:Barbarroja|Pearl Harbor|mundial))',
            "cambio": r'(194[2-3].*?(?:Stalingrado|Alamein|cambio))',
            "final": r'(194[4-5].*?(?:Normandía|rendición|final))'
        }
        
        for phase, pattern in phase_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                wwii_facts["war_phases"].append({
                    "phase": phase,
                    "description": matches[0][:200] + "..." if len(matches[0]) > 200 else matches[0]
                })
        
        # Extraer operaciones militares mencionadas
        military_ops = re.findall(r'\b(?:operación|batalla de?|invasión de?|bombardeo de?)\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)*\b', text, re.IGNORECASE)
        wwii_facts["military_operations"] = list(set(military_ops))
        
        return wwii_facts
    
    def _extract_causal_relationships(self, text: str) -> List[Dict[str, str]]:
        """Extrae relaciones de causa y efecto del texto"""
        relationships = []
        
        # Patrones de causalidad
        causal_patterns = [
            r'(?:debido a|por|a causa de)\s+([^.]{10,80})[^.]*(?:\.|\,)',
            r'([^.]{10,80})\s+(?:causó|provocó|llevó a|resultó en)\s+([^.]{10,80})',
            r'(?:como consecuencia de|resultado de)\s+([^.]{10,80})[^.]*(?:\.|\,)',
            r'([^.]{10,80})\s+(?:tuvo como consecuencia|dio lugar a)\s+([^.]{10,80})'
        ]
        
        for pattern in causal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    relationships.append({
                        "cause": match[0].strip(),
                        "effect": match[1].strip()
                    })
                elif isinstance(match, str):
                    relationships.append({
                        "element": match.strip(),
                        "type": "causal_element"
                    })
        
        return relationships[:5]  # Limitar a 5 relaciones
    
    def _extract_specific_facts(self, text: str) -> List[Dict[str, str]]:
        """Extrae hechos específicos y verificables del texto"""
        facts = []
        
        # Dividir en oraciones
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 30]
        
        # Buscar oraciones que contengan información factual específica
        factual_indicators = [
            r'\b(?:fue|era|se desarrolló|ocurrió|comenzó|terminó|duró)\b',
            r'\b(?:principalmente|especialmente|particularmente)\b',
            r'\b(?:primera|segunda|última|mayor|menor)\b',
            r'\b(?:aproximadamente|cerca de|más de|menos de)\s+\d+',
            r'\b(?:entre|desde|hasta|durante)\s+\d{4}'
        ]
        
        for sentence in sentences:
            factual_score = 0
            
            # Contar indicadores factuales
            for indicator in factual_indicators:
                if re.search(indicator, sentence, re.IGNORECASE):
                    factual_score += 1
            
            # Si tiene suficientes indicadores factuales, incluir
            if factual_score >= 2:
                facts.append({
                    "fact": sentence,
                    "score": factual_score,
                    "keywords": self._extract_keywords_from_sentence(sentence)
                })
        
        # Ordenar por puntuación y retornar los mejores
        facts.sort(key=lambda x: x["score"], reverse=True)
        return facts[:8]
    
    def _extract_keywords_from_sentence(self, sentence: str) -> List[str]:
        """Extrae palabras clave de una oración"""
        # Palabras importantes (no stop words)
        important_words = re.findall(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{3,}\b', sentence)
        
        # Filtrar stop words
        stop_words = {'Este', 'Esta', 'Estos', 'Estas', 'También', 'Durante', 'Después', 'Antes'}
        keywords = [word for word in important_words if word not in stop_words]
        
        return keywords[:4]
    
    def _generate_category_questions(self, category: str, factual_data: Dict, 
                                   analysis: Dict, difficulty: str, max_questions: int) -> List[Dict]:
        """Genera preguntas para una categoría específica"""
        questions = []
        templates = self.question_templates.get(category, [])
        
        if category == "cronologia" and factual_data.get("dates"):
            questions.extend(self._generate_chronology_questions(factual_data, templates, difficulty))
        
        elif category == "personajes" and factual_data.get("people"):
            questions.extend(self._generate_people_questions(factual_data, templates, difficulty))
        
        elif category == "geografia" and factual_data.get("places"):
            questions.extend(self._generate_geography_questions(factual_data, templates, difficulty))
        
        elif category == "causas_consecuencias" and factual_data.get("relationships"):
            questions.extend(self._generate_causality_questions(factual_data, templates, difficulty))
        
        elif category == "conceptos":
            questions.extend(self._generate_concept_questions(factual_data, analysis, templates, difficulty))
        
        return questions[:max_questions]
    
    def _generate_chronology_questions(self, factual_data: Dict, templates: List[str], difficulty: str) -> List[Dict]:
        """Genera preguntas de cronología específicas"""
        questions = []
        dates_info = factual_data.get("dates", [])
        
        for date_info in dates_info[:2]:  # Máximo 2 preguntas de cronología
            date = date_info["date"]
            context = date_info["context"]
            
            # Extraer evento del contexto
            event = self._extract_event_from_context(context)
            
            question_text = f"¿En qué año {event}?"
            
            # Crear opciones
            correct_year = int(date)
            options = [
                str(correct_year),
                str(correct_year - 1),
                str(correct_year + 1),
                str(correct_year - 2) if correct_year > 1940 else str(correct_year + 2)
            ]
            random.shuffle(options)
            correct_answer = options.index(str(correct_year))
            
            explanation = f"Según el texto, esto ocurrió en {date}. {context[:100]}..."
            
            questions.append({
                "id": len(questions) + 1,
                "question": question_text,
                "options": options,
                "correct_answer": correct_answer,
                "explanation": explanation,
                "difficulty": difficulty,
                "category": "cronologia",
                "source_context": context
            })
        
        return questions
    
    def _extract_event_from_context(self, context: str) -> str:
        """Extrae descripción del evento del contexto"""
        # Buscar patrones de eventos
        event_patterns = [
            r'(invasión de [A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)',
            r'(batalla de [A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)',
            r'(operación [A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)',
            r'(ataque a [A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)',
            r'(comenzó la [a-záéíóúüñ ]+)',
            r'(se inició [a-záéíóúüñ ]+)'
        ]
        
        for pattern in event_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        # Fallback: usar contexto general
        words = context.split()
        if len(words) > 10:
            return " ".join(words[5:10]).lower()
        
        return "ocurrió este evento"
    
    def _generate_direct_content_question(self, text: str, factual_data: Dict, 
                                        question_id: int, difficulty: str) -> Optional[Dict]:
        """Genera pregunta directa basada en contenido específico del texto"""
        
        # Usar hechos específicos extraídos
        specific_facts = factual_data.get("specific_facts", [])
        
        if not specific_facts:
            return None
        
        # Seleccionar hecho aleatorio
        fact_info = random.choice(specific_facts)
        fact_text = fact_info["fact"]
        keywords = fact_info["keywords"]
        
        # Crear pregunta basada en el hecho
        if keywords:
            main_keyword = keywords[0]
            question_text = f"Según el texto, ¿qué información es correcta sobre {main_keyword.lower()}?"
        else:
            question_text = "¿Cuál de las siguientes afirmaciones es correcta según el texto?"
        
        # Crear opciones
        correct_option = fact_text[:100] + "..." if len(fact_text) > 100 else fact_text
        
        # Generar distractores inteligentes
        distractors = self._generate_intelligent_distractors(fact_text, keywords, text)
        
        options = [correct_option] + distractors[:3]
        random.shuffle(options)
        correct_answer = options.index(correct_option)
        
        explanation = f"Esta información se encuentra directamente en el texto: '{fact_text[:150]}...'"
        
        return {
            "id": question_id,
            "question": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": explanation,
            "difficulty": difficulty,
            "category": "contenido_directo",
            "source_fact": fact_text
        }
    
    def _generate_intelligent_distractors(self, correct_fact: str, keywords: List[str], full_text: str) -> List[str]:
        """Genera distractores inteligentes pero incorrectos"""
        distractors = []
        
        # Tipo 1: Modificar fechas/números en la respuesta correcta
        modified_fact = correct_fact
        for year in re.findall(r'\b\d{4}\b', correct_fact):
            wrong_year = str(int(year) + random.choice([-1, 1, -2, 2]))
            modified_fact = modified_fact.replace(year, wrong_year)
        
        if modified_fact != correct_fact:
            distractors.append(modified_fact[:100] + "..." if len(modified_fact) > 100 else modified_fact)
        
        # Tipo 2: Usar información de otras partes del texto pero en contexto incorrecto
        other_sentences = [s.strip() for s in re.split(r'[.!?]+', full_text) 
                          if len(s.strip()) > 30 and s.strip() != correct_fact.strip()]
        
        if other_sentences:
            distractor_sentence = random.choice(other_sentences)
            distractors.append(distractor_sentence[:100] + "..." if len(distractor_sentence) > 100 else distractor_sentence)
        
        # Tipo 3: Crear afirmaciones falsas pero plausibles
        generic_distractors = [
            "Una afirmación que contradice la información presentada en el texto",
            "Una interpretación que no tiene fundamento en el contenido analizado",
            "Una conclusión que no está respaldada por las fuentes históricas mencionadas"
        ]
        
        while len(distractors) < 3:
            if generic_distractors:
                distractors.append(generic_distractors.pop(0))
            else:
                distractors.append("Una información que no aparece en el texto")
        
        return distractors
    
    def _validate_and_improve_questions(self, questions: List[Dict], original_text: str) -> List[Dict]:
        """Valida y mejora la calidad de las preguntas generadas"""
        improved_questions = []
        
        for question in questions:
            # Validar que la pregunta tenga sentido
            if self._is_valid_question(question, original_text):
                # Mejorar opciones si es necesario
                improved_question = self._improve_question_quality(question, original_text)
                improved_questions.append(improved_question)
        
        return improved_questions
    
    def _is_valid_question(self, question: Dict, text: str) -> bool:
        """Valida si una pregunta es de buena calidad"""
        
        # Verificar que la pregunta no esté vacía
        if not question.get("question") or len(question["question"]) < 10:
            return False
        
        # Verificar que tenga opciones válidas
        options = question.get("options", [])
        if len(options) != 4:
            return False
        
        # Verificar que las opciones no sean todas iguales
        if len(set(options)) < 4:
            return False
        
        # Verificar que la respuesta correcta esté dentro del rango
        correct_answer = question.get("correct_answer")
        if correct_answer is None or correct_answer < 0 or correct_answer >= len(options):
            return False
        
        return True
    
    def _improve_question_quality(self, question: Dict, text: str) -> Dict:
        """Mejora la calidad de una pregunta específica"""
        
        # Mejorar la formulación de la pregunta
        question_text = question["question"]
        
        # Asegurar que empiece con ¿ y termine con ?
        if not question_text.startswith("¿"):
            question_text = "¿" + question_text.capitalize()
        if not question_text.endswith("?"):
            question_text += "?"
        
        question["question"] = question_text
        
        # Mejorar explicación si es muy corta
        explanation = question.get("explanation", "")
        if len(explanation) < 50:
            question["explanation"] = self._generate_better_explanation(question, text)
        
        return question
    
    def _generate_better_explanation(self, question: Dict, text: str) -> str:
        """Genera una explicación mejorada para la pregunta"""
        
        correct_option = question["options"][question["correct_answer"]]
        category = question.get("category", "general")
        
        explanations_by_category = {
            "cronologia": f"La respuesta correcta es '{correct_option[:50]}...' porque esta fecha se menciona específicamente en el texto y corresponde al período histórico analizado.",
            "personajes": f"La respuesta correcta es '{correct_option[:50]}...' porque el texto proporciona información específica sobre este personaje y su papel en los eventos descritos.",
            "geografia": f"La respuesta correcta es '{correct_option[:50]}...' porque el texto hace referencia específica a este lugar y su importancia en el contexto histórico.",
            "causas_consecuencias": f"La respuesta correcta es '{correct_option[:50]}...' porque el texto establece claramente esta relación causal entre los eventos descritos.",
            "conceptos": f"La respuesta correcta es '{correct_option[:50]}...' porque el texto define y explica este concepto en el contexto del tema analizado.",
            "contenido_directo": f"La respuesta correcta es '{correct_option[:50]}...' porque esta información se encuentra directamente en el contenido del texto analizado."
        }
        
        return explanations_by_category.get(category, 
                                          f"La respuesta correcta se basa en la información específica proporcionada en el texto: '{correct_option[:50]}...'")
    
    def generate_quiz_with_enhanced_context(self, text: str, key_concepts: List[str], 
                                          num_questions: int = 5, difficulty: str = "medium") -> Dict[str, Any]:
        """
        Función principal mejorada que integra generación contextual con conceptos clave
        """
        logger.info(f"🚀 Generando quiz contextual mejorado con {num_questions} preguntas")
        
        try:
            # Generar preguntas contextuales
            contextual_questions = self.generate_contextual_quiz(text, num_questions, difficulty)
            
            # Si no se generaron suficientes preguntas contextuales, completar con conceptos clave
            if len(contextual_questions) < num_questions:
                remaining = num_questions - len(contextual_questions)
                concept_questions = self._generate_concept_based_questions(
                    text, key_concepts, remaining, difficulty
                )
                contextual_questions.extend(concept_questions)
            
            # Asegurar IDs únicos
            for i, question in enumerate(contextual_questions):
                question["id"] = i + 1
            
            result = {
                "questions": contextual_questions[:num_questions],
                "success": True,
                "generation_method": "enhanced_contextual",
                "content_analysis": "specific_content_extraction"
            }
            
            logger.info(f"✅ Quiz contextual generado exitosamente con {len(result['questions'])} preguntas")
            return result
            
        except Exception as e:
            logger.error(f"Error generando quiz contextual: {e}")
            # Fallback a generación básica
            return self._generate_fallback_quiz(text, key_concepts, num_questions, difficulty)
    
    def _generate_concept_based_questions(self, text: str, concepts: List[str], 
                                        num_questions: int, difficulty: str) -> List[Dict]:
        """Genera preguntas basadas en conceptos clave extraídos"""
        questions = []
        
        for i, concept in enumerate(concepts[:num_questions]):
            # Buscar contexto del concepto en el texto
            concept_context = self._find_concept_context(concept, text)
            
            # Crear pregunta específica del concepto
            question = self._create_concept_question(concept, concept_context, i + 1, difficulty)
            if question:
                questions.append(question)
        
        return questions
    
    def _find_concept_context(self, concept: str, text: str) -> str:
        """Encuentra el contexto específico donde aparece un concepto"""
        
        # Buscar oraciones que contengan el concepto
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if concept.lower() in sentence.lower() and len(sentence.strip()) > 20:
                return sentence.strip()
        
        # Si no se encuentra contexto específico, retornar una porción del texto
        return text[:200] + "..."
    
    def _create_concept_question(self, concept: str, context: str, 
                               question_id: int, difficulty: str) -> Optional[Dict]:
        """Crea pregunta específica basada en un concepto y su contexto"""
        
        # Plantillas específicas para conceptos
        concept_templates = [
            f"Según el texto, ¿cuál es la característica principal de {concept}?",
            f"¿Cómo se describe {concept} en el contenido analizado?",
            f"¿Qué papel desempeña {concept} según la información presentada?",
            f"¿Cuál es la importancia de {concept} en el contexto del tema?",
            f"Según el texto, ¿qué aspectos definen a {concept}?"
        ]
        
        question_text = random.choice(concept_templates)
        
        # Crear respuesta correcta basada en el contexto
        if len(context) > 50:
            correct_answer = context[:80] + "..." if len(context) > 80 else context
        else:
            correct_answer = f"Lo que se describe específicamente en el texto sobre {concept}"
        
        # Crear distractores
        distractors = [
            f"Una característica que no se menciona en el texto sobre {concept}",
            f"Una interpretación incorrecta del rol de {concept}",
            f"Una información que contradice lo expuesto sobre {concept}"
        ]
        
        # Combinar opciones
        options = [correct_answer] + distractors
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        
        explanation = f"La respuesta correcta se basa en la información específica que el texto proporciona sobre {concept}: '{context[:100]}...'"
        
        return {
            "id": question_id,
            "question": question_text,
            "options": options,
            "correct_answer": correct_index,
            "explanation": explanation,
            "difficulty": difficulty,
            "category": "concepto_clave",
            "concept": concept
        }
    
    def _generate_fallback_quiz(self, text: str, concepts: List[str], 
                              num_questions: int, difficulty: str) -> Dict[str, Any]:
        """Genera quiz de respaldo cuando falla el método principal"""
        
        logger.warning("Usando generación de quiz de respaldo")
        
        questions = []
        
        # Usar método simplificado pero funcional
        for i in range(num_questions):
            concept = concepts[i % len(concepts)] if concepts else f"concepto {i+1}"
            
            question = {
                "id": i + 1,
                "question": f"¿Cuál es la información más relevante sobre {concept} según el texto?",
                "options": [
                    f"La información específica que se presenta sobre {concept} en el contenido",
                    f"Una interpretación que no está en el texto sobre {concept}",
                    f"Una conclusión no respaldada por el contenido sobre {concept}",
                    f"Una información contradictoria al texto sobre {concept}"
                ],
                "correct_answer": 0,
                "explanation": f"La respuesta se basa en la información específica que el texto proporciona sobre {concept}.",
                "difficulty": difficulty,
                "category": "fallback"
            }
            
            questions.append(question)
        
        return {
            "questions": questions,
            "success": True,
            "generation_method": "fallback",
            "note": "Generación simplificada por problemas en el método principal"
        }

# Función auxiliar para calcular relevancia
def calculate_question_relevance(question: Dict, text: str, concepts: List[str]) -> float:
    """Calcula la relevancia de una pregunta basada en el contenido"""
    
    relevance_score = 0.0
    question_text = question.get("question", "").lower()
    
    # Puntos por mencionar conceptos clave
    for concept in concepts:
        if concept.lower() in question_text:
            relevance_score += 0.3
    
    # Puntos por categoría específica
    category = question.get("category", "")
    if category in ["cronologia", "personajes", "contenido_directo"]:
        relevance_score += 0.4
    
    # Puntos por tener contexto fuente
    if question.get("source_context") or question.get("source_fact"):
        relevance_score += 0.3
    
    return min(relevance_score, 1.0)

# Función para mejorar preguntas existentes
def enhance_existing_questions(questions: List[Dict], text: str, concepts: List[str]) -> List[Dict]:
    """Mejora preguntas existentes con información contextual"""
    
    enhanced_questions = []
    
    for question in questions:
        enhanced_question = question.copy()
        
        # Mejorar explicación con contexto específico
        if len(enhanced_question.get("explanation", "")) < 100:
            enhanced_question["explanation"] = generate_contextual_explanation(
                enhanced_question, text, concepts
            )
        
        # Mejorar opciones para que sean más específicas
        enhanced_question["options"] = improve_question_options(
            enhanced_question["options"], text
        )
        
        enhanced_questions.append(enhanced_question)
    
    return enhanced_questions

def generate_contextual_explanation(question: Dict, text: str, concepts: List[str]) -> str:
    """Genera explicación contextual para una pregunta"""
    
    correct_option = question["options"][question["correct_answer"]]
    
    # Buscar evidencia en el texto
    text_lower = text.lower()
    option_lower = correct_option.lower()
    
    # Buscar palabras clave de la opción en el texto
    option_words = [word for word in option_lower.split() if len(word) > 3]
    
    evidence_sentences = []
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        word_matches = sum(1 for word in option_words if word in sentence_lower)
        
        if word_matches >= 2 and len(sentence.strip()) > 30:
            evidence_sentences.append(sentence.strip())
    
    if evidence_sentences:
        evidence = evidence_sentences[0][:150] + "..."
        return f"La respuesta correcta es '{correct_option[:50]}...' porque el texto establece: '{evidence}'"
    
    return f"La respuesta correcta es '{correct_option[:50]}...' basándose en la información específica proporcionada en el contenido del texto."

def improve_question_options(options: List[str], text: str) -> List[str]:
    """Mejora las opciones de una pregunta para que sean más específicas"""
    
    improved_options = []
    
    for option in options:
        # Si la opción es muy genérica, hacerla más específica
        if "información" in option.lower() and "texto" in option.lower():
            # Intentar hacer la opción más específica usando información del texto
            improved_option = make_option_more_specific(option, text)
            improved_options.append(improved_option)
        else:
            improved_options.append(option)
    
    return improved_options

def make_option_more_specific(generic_option: str, text: str) -> str:
    """Hace una opción genérica más específica basada en el texto"""
    
    # Extraer información específica del texto
    specific_elements = []
    
    # Buscar fechas
    dates = re.findall(r'\b\d{4}\b', text)
    if dates:
        specific_elements.extend(dates[:2])
    
    # Buscar lugares
    places = re.findall(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)*\b', text)
    if places:
        specific_elements.extend(places[:2])
    
    if specific_elements:
        element = random.choice(specific_elements)
        return generic_option.replace("información", f"información sobre {element}")
    
    return generic_option