# app/services/simple_contextual_quiz.py - GENERADOR SIMPLE PERO EFECTIVO
import re
import random
import logging
from typing import Dict, List, Any, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

class SimpleContextualQuizGenerator:
    """
    Generador simple pero efectivo que crea preguntas específicas del contenido
    """
    
    def __init__(self):
        self._setup_patterns()
        
    def _setup_patterns(self):
        """Configura patrones de extracción de información"""
        
        # Patrones específicos para Segunda Guerra Mundial
        self.wwii_patterns = {
            "fechas": r'\b(19[3-4]\d)\b',
            "eventos": r'\b(invasión de \w+|batalla de \w+|operación \w+|bombardeo de \w+|ataque a \w+)\b',
            "personajes": r'\b(Hitler|Stalin|Roosevelt|Churchill|Mussolini|Franco|De Gaulle)\b',
            "paises": r'\b(Alemania|Francia|Italia|España|Polonia|URSS|Estados Unidos|Reino Unido|Japón)\b',
            "conceptos": r'\b(Blitzkrieg|Holocausto|fascismo|nazismo|democracia|totalitarismo)\b',
            "organizaciones": r'\b(Wehrmacht|SS|Gestapo|Luftwaffe|Aliados|Eje)\b'
        }
        
        # Patrones generales para otros temas
        self.general_patterns = {
            "fechas": r'\b(20\d{2}|19\d{2}|\d{1,2} de \w+ de \d{4})\b',
            "numeros": r'\b(\d+(?:\.\d+)?)\s*(?:%|por ciento|millones?|miles?)\b',
            "procesos": r'\b(\w+ción|\w+miento|\w+aje)\b',
            "conceptos_importantes": r'\b([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{4,})\b'
        }
    
    def generate_contextual_quiz(self, text: str, num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Genera quiz contextual específico del texto
        """
        logger.info(f"🎯 Generando {num_questions} preguntas contextuales específicas")
        
        # 1. Analizar el contenido del texto
        content_info = self._extract_content_information(text)
        
        # 2. Generar preguntas específicas
        questions = []
        
        # Priorizar tipos de preguntas según el contenido disponible
        if content_info["domain"] == "historia_wwii":
            questions.extend(self._generate_wwii_questions(content_info, text))
        else:
            questions.extend(self._generate_general_questions(content_info, text))
        
        # 3. Completar con preguntas de hechos específicos si faltan
        while len(questions) < num_questions:
            fact_question = self._generate_fact_based_question(text, len(questions) + 1)
            if fact_question:
                questions.append(fact_question)
            else:
                break
        
        # 4. Asegurar IDs únicos y validar
        validated_questions = []
        for i, question in enumerate(questions[:num_questions]):
            question["id"] = i + 1
            if self._is_question_valid(question, text):
                validated_questions.append(question)
        
        logger.info(f"✅ Generadas {len(validated_questions)} preguntas contextuales válidas")
        return validated_questions
    
    def _extract_content_information(self, text: str) -> Dict[str, Any]:
        """
        Extrae información específica del contenido
        """
        info = {
            "domain": "general",
            "fechas": [],
            "eventos": [],
            "personajes": [],
            "lugares": [],
            "conceptos": [],
            "hechos_especificos": []
        }
        
        text_lower = text.lower()
        
        # Detectar dominio específico
        if "segunda guerra mundial" in text_lower or ("hitler" in text_lower and "1939" in text):
            info["domain"] = "historia_wwii"
            patterns = self.wwii_patterns
        else:
            patterns = self.general_patterns
        
        # Extraer información usando patrones
        for category, pattern in patterns.items():
            matches = list(set(re.findall(pattern, text, re.IGNORECASE)))
            if matches and category in info:
                info[category] = matches[:5]  # Limitar a 5 elementos por categoría
        
        # Extraer hechos específicos (oraciones con información concreta)
        info["hechos_especificos"] = self._extract_specific_facts(text)
        
        logger.info(f"Dominio detectado: {info['domain']}")
        logger.info(f"Información extraída: {len(info['hechos_especificos'])} hechos específicos")
        
        return info
    
    def _extract_specific_facts(self, text: str) -> List[Dict[str, str]]:
        """
        Extrae hechos específicos del texto
        """
        facts = []
        
        # Dividir en oraciones
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 30]
        
        for sentence in sentences:
            # Buscar oraciones con información factual específica
            factual_indicators = [
                r'\b\d{4}\b',  # Contiene años
                r'\b(comenzó|terminó|inició|finalizó|ocurrió|se desarrolló)\b',  # Verbos de eventos
                r'\b(fue|era|es|son|fueron)\b',  # Verbos de estado
                r'\b(primera|segundo|último|principal|mayor|menor)\b',  # Adjetivos ordinales
                r'\b(causa|consecuencia|resultado|efecto|debido a|por)\b'  # Causalidad
            ]
            
            factual_score = 0
            for indicator in factual_indicators:
                if re.search(indicator, sentence, re.IGNORECASE):
                    factual_score += 1
            
            # Si tiene suficientes indicadores factuales
            if factual_score >= 2:
                # Extraer elemento principal de la oración
                main_subject = self._extract_main_subject(sentence)
                
                facts.append({
                    "sentence": sentence,
                    "subject": main_subject,
                    "score": factual_score
                })
        
        # Ordenar por puntuación y retornar los mejores
        facts.sort(key=lambda x: x["score"], reverse=True)
        return facts[:10]
    
    def _extract_main_subject(self, sentence: str) -> str:
        """
        Extrae el sujeto principal de una oración
        """
        # Buscar patrones de sujetos principales
        subject_patterns = [
            r'^([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ\s]+)',  # Inicio de oración con mayúscula
            r'\b(La\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ\s]+)',  # "La Segunda Guerra..."
            r'\b(El\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ\s]+)',   # "El conflicto..."
        ]
        
        for pattern in subject_patterns:
            match = re.search(pattern, sentence)
            if match:
                subject = match.group(1).strip()
                if len(subject) < 50:  # No muy largo
                    return subject
        
        # Fallback: usar las primeras palabras
        words = sentence.split()[:4]
        return " ".join(words)
    
    def _generate_wwii_questions(self, content_info: Dict, text: str) -> List[Dict[str, Any]]:
        """
        Genera preguntas específicas sobre la Segunda Guerra Mundial
        """
        questions = []
        
        # 1. Pregunta sobre fechas si hay información temporal
        if content_info.get("fechas"):
            fecha_question = self._create_date_question(content_info["fechas"], text)
            if fecha_question:
                questions.append(fecha_question)
        
        # 2. Pregunta sobre eventos específicos
        if content_info.get("eventos"):
            evento_question = self._create_event_question(content_info["eventos"], text)
            if evento_question:
                questions.append(evento_question)
        
        # 3. Pregunta sobre personajes
        if content_info.get("personajes"):
            personaje_question = self._create_character_question(content_info["personajes"], text)
            if personaje_question:
                questions.append(personaje_question)
        
        # 4. Pregunta sobre conceptos específicos
        if content_info.get("conceptos"):
            concepto_question = self._create_concept_question(content_info["conceptos"], text)
            if concepto_question:
                questions.append(concepto_question)
        
        return questions
    
    def _create_date_question(self, fechas: List[str], text: str) -> Dict[str, Any]:
        """
        Crea pregunta específica sobre fechas
        """
        fecha_principal = fechas[0]  # Tomar la primera fecha encontrada
        
        # Buscar contexto de la fecha en el texto
        contexto = self._find_date_context(fecha_principal, text)
        
        # Crear pregunta específica
        if "1939" in fecha_principal:
            question_text = "¿En qué año comenzó la Segunda Guerra Mundial?"
            correct_answer = "1939"
        elif "1945" in fecha_principal:
            question_text = "¿En qué año terminó la Segunda Guerra Mundial?"
            correct_answer = "1945"
        else:
            question_text = f"¿En qué año ocurrió el evento mencionado en el texto?"
            correct_answer = fecha_principal
        
        # Crear opciones
        base_year = int(fecha_principal)
        options = [
            correct_answer,
            str(base_year - 1),
            str(base_year + 1),
            str(base_year - 2) if base_year > 1940 else str(base_year + 2)
        ]
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        
        explanation = f"Según el texto, esto ocurrió en {fecha_principal}. {contexto[:100]}..."
        
        return {
            "id": 1,
            "question": question_text,
            "options": options,
            "correct_answer": correct_index,
            "explanation": explanation,
            "difficulty": "medium",
            "category": "cronologia",
            "source": "fecha_especifica"
        }
    
    def _find_date_context(self, fecha: str, text: str) -> str:
        """
        Encuentra el contexto donde aparece una fecha específica
        """
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if fecha in sentence and len(sentence.strip()) > 20:
                return sentence.strip()
        
        return f"El año {fecha} es mencionado en el contexto del texto"
    
    def _create_event_question(self, eventos: List[str], text: str) -> Dict[str, Any]:
        """
        Crea pregunta específica sobre eventos
        """
        evento_principal = eventos[0]
        
        # Crear pregunta específica según el evento
        if "invasión" in evento_principal.lower():
            question_text = f"¿Qué evento específico se menciona en el texto?"
            correct_answer = evento_principal.title()
        elif "batalla" in evento_principal.lower():
            question_text = f"¿Qué batalla específica se describe en el contenido?"
            correct_answer = evento_principal.title()
        else:
            question_text = f"¿Cuál de los siguientes eventos se menciona en el texto?"
            correct_answer = evento_principal.title()
        
        # Crear distractores realistas pero incorrectos
        distractores = [
            "Batalla de Waterloo",
            "Invasión de Normandía (si no es el evento correcto)",
            "Operación Market Garden",
            "Bombardeo de Londres"
        ]
        
        # Seleccionar 3 distractores que no sean el evento correcto
        distractores_finales = [d for d in distractores if d.lower() not in correct_answer.lower()][:3]
        
        options = [correct_answer] + distractores_finales
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        
        contexto = self._find_event_context(evento_principal, text)
        explanation = f"El texto menciona específicamente: {evento_principal}. {contexto[:100]}..."
        
        return {
            "id": 2,
            "question": question_text,
            "options": options,
            "correct_answer": correct_index,
            "explanation": explanation,
            "difficulty": "medium",
            "category": "eventos",
            "source": "evento_especifico"
        }
    
    def _find_event_context(self, evento: str, text: str) -> str:
        """
        Encuentra el contexto de un evento específico
        """
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if evento.lower() in sentence.lower() and len(sentence.strip()) > 20:
                return sentence.strip()
        
        return f"El evento {evento} es mencionado en el contexto histórico del texto"
    
    def _create_character_question(self, personajes: List[str], text: str) -> Dict[str, Any]:
        """
        Crea pregunta específica sobre personajes históricos
        """
        personaje_principal = personajes[0]
        
        question_text = f"¿Qué personaje histórico se menciona como relevante en el texto?"
        correct_answer = personaje_principal
        
        # Distractores de personajes históricos pero que no aparecen en el texto
        otros_personajes = ["Napoleon Bonaparte", "Winston Churchill", "Franklin Roosevelt", "José Stalin", "Benito Mussolini"]
        distractores = [p for p in otros_personajes if p != personaje_principal][:3]
        
        options = [correct_answer] + distractores
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        
        contexto = self._find_character_context(personaje_principal, text)
        explanation = f"El texto menciona específicamente a {personaje_principal}. {contexto[:100]}..."
        
        return {
            "id": 3,
            "question": question_text,
            "options": options,
            "correct_answer": correct_index,
            "explanation": explanation,
            "difficulty": "medium",
            "category": "personajes",
            "source": "personaje_especifico"
        }
    
    def _find_character_context(self, personaje: str, text: str) -> str:
        """
        Encuentra el contexto donde se menciona un personaje
        """
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if personaje.lower() in sentence.lower() and len(sentence.strip()) > 20:
                return sentence.strip()
        
        return f"{personaje} es mencionado en el contexto histórico del texto"
    
    def _create_concept_question(self, conceptos: List[str], text: str) -> Dict[str, Any]:
        """
        Crea pregunta específica sobre conceptos
        """
        concepto_principal = conceptos[0]
        
        question_text = f"¿Cómo se describe {concepto_principal} en el texto?"
        
        # Buscar descripción específica del concepto en el texto
        descripcion = self._find_concept_description(concepto_principal, text)
        
        if descripcion:
            correct_answer = descripcion[:80] + "..." if len(descripcion) > 80 else descripcion
        else:
            correct_answer = f"Como un elemento importante en el contexto histórico analizado"
        
        # Crear distractores
        distractores = [
            f"Como un factor secundario sin mayor relevancia",
            f"Como una teoría no comprobada históricamente",  
            f"Como un elemento que no influyó en los eventos principales"
        ]
        
        options = [correct_answer] + distractores
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        
        explanation = f"El texto describe {concepto_principal} de manera específica en el contexto del tema analizado."
        
        return {
            "id": 4,
            "question": question_text,
            "options": options,
            "correct_answer": correct_index,
            "explanation": explanation,
            "difficulty": "medium",
            "category": "conceptos",
            "source": "concepto_especifico"
        }
    
    def _find_concept_description(self, concepto: str, text: str) -> str:
        """
        Encuentra la descripción específica de un concepto en el texto
        """
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if concepto.lower() in sentence.lower() and len(sentence.strip()) > 30:
                # Buscar la parte de la oración que describe el concepto
                if "es" in sentence or "fue" in sentence or "era" in sentence:
                    return sentence.strip()
        
        return ""
    
    def _generate_fact_based_question(self, text: str, question_id: int) -> Dict[str, Any]:
        """
        Genera pregunta basada en hechos específicos del texto
        """
        # Extraer hechos específicos si no se ha hecho antes
        facts = self._extract_specific_facts(text)
        
        if not facts:
            return None
        
        # Seleccionar un hecho aleatorio
        fact = random.choice(facts)
        fact_sentence = fact["sentence"]
        subject = fact["subject"]
        
        question_text = f"Según el texto, ¿qué información es correcta sobre {subject.lower()}?"
        
        # Usar la oración completa como respuesta correcta (acortada si es necesaria)
        correct_answer = fact_sentence[:100] + "..." if len(fact_sentence) > 100 else fact_sentence
        
        # Crear distractores genéricos pero específicos
        distractores = [
            f"Una característica que no se menciona en el texto sobre {subject.lower()}",
            f"Una interpretación incorrecta de los hechos sobre {subject.lower()}",
            f"Una información que contradice lo expuesto sobre {subject.lower()}"
        ]
        
        options = [correct_answer] + distractores
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        
        explanation = f"Esta información se encuentra directamente en el texto: '{fact_sentence[:120]}...'"
        
        return {
            "id": question_id,
            "question": question_text,
            "options": options,
            "correct_answer": correct_index,
            "explanation": explanation,
            "difficulty": "medium",
            "category": "hecho_especifico",
            "source": "oracion_textual"
        }
    
    def _generate_general_questions(self, content_info: Dict, text: str) -> List[Dict[str, Any]]:
        """
        Genera preguntas para contenido general (no específico de WWII)
        """
        questions = []
        
        # Usar hechos específicos como base para preguntas generales
        facts = content_info.get("hechos_especificos", [])
        
        for i, fact in enumerate(facts[:3]):  # Máximo 3 preguntas de hechos
            question = self._create_general_fact_question(fact, i + 1)
            if question:
                questions.append(question)
        
        return questions
    
    def _create_general_fact_question(self, fact: Dict, question_id: int) -> Dict[str, Any]:
        """
        Crea pregunta general basada en un hecho específico
        """
        fact_sentence = fact["sentence"]
        subject = fact["subject"]
        
        question_text = f"Según el texto analizado, ¿cuál es la información correcta?"
        
        correct_answer = fact_sentence[:90] + "..." if len(fact_sentence) > 90 else fact_sentence
        
        distractores = [
            "Una afirmación que no se encuentra en el contenido analizado",
            "Una interpretación que contradice la información presentada",
            "Una conclusión que no está respaldada por el texto"
        ]
        
        options = [correct_answer] + distractores
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        
        explanation = f"Esta información aparece específicamente en el texto: '{fact_sentence[:100]}...'"
        
        return {
            "id": question_id,
            "question": question_text,
            "options": options,
            "correct_answer": correct_index,
            "explanation": explanation,
            "difficulty": "medium",
            "category": "contenido_general",
            "source": "hecho_textual"
        }
    
    def _is_question_valid(self, question: Dict, text: str) -> bool:
        """
        Valida que una pregunta sea de buena calidad
        """
        # Verificaciones básicas
        if not question.get("question") or len(question["question"]) < 10:
            return False
        
        options = question.get("options", [])
        if len(options) != 4:
            return False
        
        correct_answer = question.get("correct_answer")
        if correct_answer is None or correct_answer < 0 or correct_answer >= 4:
            return False
        
        # Verificar que las opciones no sean todas iguales
        if len(set(options)) < 4:
            return False
        
        return True