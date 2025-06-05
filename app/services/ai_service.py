# app/services/ai_service.py - VERSIÓN MEJORADA PARA MEJORES PREGUNTAS
import json
import logging
import random
import re
from typing import Dict, Any, List, Optional
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    pipeline, T5ForConditionalGeneration, T5Tokenizer
)
import torch
from app.core.config import settings

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Inicializar modelos
        self._init_models()
    
    def _init_models(self):
        """Inicializa todos los modelos necesarios"""
        try:
            # Modelo para resúmenes (BART en español)
            logger.info("Cargando modelo de resumen...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1
            )
            
            # Modelo para generación de texto/quiz (T5)
            logger.info("Cargando modelo T5 para quiz...")
            self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                "google/flan-t5-base"
            ).to(self.device)
            
            # Pipeline para clasificación y análisis
            self.classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Todos los modelos cargados exitosamente")
            
        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")
            self._init_fallback_models()
    
    def _init_fallback_models(self):
        """Modelos de respaldo más pequeños"""
        logger.info("Cargando modelos de respaldo...")
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    
    async def generate_summary(self, text: str, length: str = "medium") -> Dict[str, Any]:
        """
        Genera un resumen educativo del texto usando BART
        """
        try:
            # Configurar longitud según parámetro
            length_config = {
                "short": {"max_length": 100, "min_length": 30},
                "medium": {"max_length": 200, "min_length": 50},
                "long": {"max_length": 300, "min_length": 100}
            }
            
            config = length_config.get(length, length_config["medium"])
            
            # Limitar texto de entrada (BART tiene límite de tokens)
            max_input_length = 1024
            if len(text.split()) > max_input_length:
                text = " ".join(text.split()[:max_input_length])
            
            # Generar resumen
            summary_result = self.summarizer(
                text,
                max_length=config["max_length"],
                min_length=config["min_length"],
                do_sample=False
            )
            
            summary = summary_result[0]['summary_text']
            
            # Post-procesar para hacerlo más educativo
            educational_summary = self._make_educational(summary, text)
            
            return {
                "summary": educational_summary,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generando resumen: {e}")
            return {
                "summary": self._generate_fallback_summary(text),
                "success": False,
                "error": str(e)
            }
    
    def _make_educational(self, summary: str, original_text: str) -> str:
        """Mejora el resumen para hacerlo más educativo"""
        # Agregar contexto educativo
        intro = "📚 **Resumen Educativo:**\n\n"
        
        # Identificar conceptos clave del texto original
        key_concepts = self._extract_key_terms(original_text)
        
        if key_concepts:
            intro += f"🔑 **Conceptos clave:** {', '.join(key_concepts[:3])}\n\n"
        
        return intro + summary
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extrae términos clave del texto de manera más inteligente"""
        # Limpiar texto
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()
        
        # Stop words más completas
        stop_words = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 
            'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 
            'las', 'una', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 
            'si', 'está', 'han', 'ser', 'muy', 'puede', 'tiene', 'entre', 'todo',
            'también', 'cuando', 'donde', 'cual', 'cada', 'desde', 'sobre', 'hasta'
        }
        
        # Filtrar palabras significativas (mínimo 4 caracteres)
        significant_words = [
            word for word in words 
            if len(word) >= 4 and word not in stop_words
        ]
        
        # Contar frecuencias
        from collections import Counter
        word_freq = Counter(significant_words)
        
        # Obtener los términos más frecuentes
        return [word.title() for word, count in word_freq.most_common(10) if count > 1]
    
    async def generate_quiz(self, text: str, key_concepts: List[str], num_questions: int = 5, difficulty: str = "medium") -> Dict[str, Any]:
        """
        Genera un quiz mejorado con preguntas más inteligentes
        """
        try:
            questions = []
            
            # ✅ EXTRAER FRASES RELEVANTES DEL TEXTO
            sentences = self._extract_meaningful_sentences(text)
            processed_concepts = self._process_concepts_for_questions(key_concepts, text)
            
            # ✅ GENERAR PREGUNTAS VARIADAS Y DE CALIDAD
            for i in range(num_questions):
                question_data = await self._generate_intelligent_question(
                    text, sentences, processed_concepts, i+1, difficulty
                )
                if question_data:
                    questions.append(question_data)
            
            # ✅ VERIFICAR CALIDAD Y COMPLETAR SI ES NECESARIO
            while len(questions) < num_questions:
                fallback_question = self._create_enhanced_fallback_question(
                    len(questions) + 1, processed_concepts, sentences, difficulty
                )
                questions.append(fallback_question)
            
            return {
                "questions": questions,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generando quiz: {e}")
            return {
                "questions": self._generate_enhanced_fallback_quiz(text, key_concepts, num_questions),
                "success": False,
                "error": str(e)
            }
    
    def _extract_meaningful_sentences(self, text: str) -> List[Dict]:
        """Extrae oraciones significativas del texto"""
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Filtrar oraciones muy cortas o muy largas
            if 20 <= len(sentence) <= 200 and len(sentence.split()) >= 5:
                meaningful_sentences.append({
                    "text": sentence,
                    "words": len(sentence.split()),
                    "concepts": self._identify_concepts_in_sentence(sentence)
                })
        
        return meaningful_sentences[:10]  # Máximo 10 oraciones
    
    def _identify_concepts_in_sentence(self, sentence: str) -> List[str]:
        """Identifica conceptos en una oración"""
        # Buscar patrones de conceptos técnicos
        patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Dos palabras capitalizadas
            r'\b[a-záéíóúüñ]+ción\b',        # Palabras terminadas en -ción
            r'\b[a-záéíóúüñ]+dad\b',         # Palabras terminadas en -dad
            r'\b[a-záéíóúüñ]+ismo\b',        # Palabras terminadas en -ismo
        ]
        
        concepts = []
        for pattern in patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            concepts.extend(matches)
        
        return list(set(concepts))[:3]  # Máximo 3 conceptos por oración
    
    def _process_concepts_for_questions(self, key_concepts: List[str], text: str) -> List[Dict]:
        """Procesa conceptos para generar mejores preguntas"""
        processed = []
        
        for concept in key_concepts[:8]:  # Limitar a 8 conceptos principales
            concept_info = {
                "name": concept,
                "context": self._find_concept_context(concept, text),
                "type": self._classify_concept_type(concept),
                "related_terms": self._find_related_terms(concept, text)
            }
            processed.append(concept_info)
        
        return processed
    
    def _find_concept_context(self, concept: str, text: str) -> str:
        """Encuentra el contexto donde aparece un concepto"""
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if concept.lower() in sentence.lower():
                return sentence.strip()
        
        return ""
    
    def _classify_concept_type(self, concept: str) -> str:
        """Clasifica el tipo de concepto"""
        concept_lower = concept.lower()
        
        if any(word in concept_lower for word in ['técnica', 'tecnología', 'método', 'algoritmo']):
            return "technical"
        elif any(word in concept_lower for word in ['proceso', 'sistema', 'modelo']):
            return "process"
        elif any(word in concept_lower for word in ['teoría', 'concepto', 'principio']):
            return "theoretical"
        else:
            return "general"
    
    def _find_related_terms(self, concept: str, text: str) -> List[str]:
        """Encuentra términos relacionados con un concepto"""
        # Buscar palabras que aparecen cerca del concepto
        words = text.lower().split()
        related = []
        
        try:
            concept_indices = [i for i, word in enumerate(words) if concept.lower() in word]
            
            for idx in concept_indices:
                # Tomar palabras en un rango de ±3 posiciones
                start = max(0, idx - 3)
                end = min(len(words), idx + 4)
                context_words = words[start:end]
                
                # Filtrar palabras significativas
                significant = [
                    w.title() for w in context_words 
                    if len(w) > 3 and w not in ['que', 'para', 'con', 'una', 'como']
                ]
                related.extend(significant)
        except:
            pass
        
        # Retornar términos únicos
        return list(set(related))[:5]
    
    async def _generate_intelligent_question(
        self, text: str, sentences: List[Dict], concepts: List[Dict], 
        question_num: int, difficulty: str
    ) -> Optional[Dict]:
        """Genera una pregunta inteligente basada en el contexto"""
        
        try:
            # Seleccionar concepto para esta pregunta
            if concepts and len(concepts) > 0:
                concept_index = (question_num - 1) % len(concepts)
                main_concept = concepts[concept_index]
            else:
                return None
            
            # Seleccionar oración relevante
            relevant_sentence = None
            for sentence in sentences:
                if main_concept["name"].lower() in sentence["text"].lower():
                    relevant_sentence = sentence
                    break
            
            if not relevant_sentence and sentences:
                relevant_sentence = sentences[min(question_num-1, len(sentences)-1)]
            
            # Generar pregunta basada en el tipo de concepto
            return self._create_context_based_question(
                main_concept, relevant_sentence, question_num, difficulty, concepts
            )
            
        except Exception as e:
            logger.error(f"Error generando pregunta inteligente {question_num}: {e}")
            return None
    
    def _create_context_based_question(
        self, concept: Dict, sentence: Dict, question_id: int, 
        difficulty: str, all_concepts: List[Dict]
    ) -> Dict:
        """Crea una pregunta basada en contexto real"""
        
        concept_name = concept["name"]
        concept_type = concept.get("type", "general")
        context = concept.get("context", "")
        
        # Plantillas de preguntas según el tipo de concepto
        question_templates = {
            "technical": [
                f"¿Cuál es la función principal de {concept_name}?",
                f"¿Cómo funciona {concept_name} según el texto?",
                f"¿Qué características tiene {concept_name}?"
            ],
            "process": [
                f"¿Qué pasos involucra {concept_name}?",
                f"¿Cuál es el objetivo de {concept_name}?",
                f"¿Qué elementos componen {concept_name}?"
            ],
            "theoretical": [
                f"¿Qué principio subyace en {concept_name}?",
                f"¿Cómo se define {concept_name} en el contexto del texto?",
                f"¿Qué importancia tiene {concept_name}?"
            ],
            "general": [
                f"¿Qué es {concept_name} según el texto?",
                f"¿Por qué es relevante {concept_name}?",
                f"¿Cómo se relaciona {concept_name} con el tema principal?"
            ]
        }
        
        # Seleccionar plantilla
        templates = question_templates.get(concept_type, question_templates["general"])
        question_text = templates[(question_id - 1) % len(templates)]
        
        # Crear opciones más inteligentes
        correct_answer = self._generate_correct_answer(concept, context)
        incorrect_options = self._generate_intelligent_distractors(concept, all_concepts, context)
        
        # Combinar opciones
        all_options = [correct_answer] + incorrect_options[:3]
        random.shuffle(all_options)
        correct_index = all_options.index(correct_answer)
        
        # Generar explicación contextual
        explanation = self._generate_contextual_explanation(concept, correct_answer, context)
        
        return {
            "id": question_id,
            "question": question_text,
            "options": all_options,
            "correct_answer": correct_index,
            "explanation": explanation,
            "difficulty": difficulty
        }
    
    def _generate_correct_answer(self, concept: Dict, context: str) -> str:
        """Genera una respuesta correcta basada en el contexto"""
        concept_name = concept["name"]
        
        # Si hay contexto, extraer información relevante
        if context:
            # Buscar definiciones o descripciones en el contexto
            context_words = context.split()
            
            # Encontrar la posición del concepto en el contexto
            try:
                concept_pos = next(i for i, word in enumerate(context_words) 
                                 if concept_name.lower() in word.lower())
                
                # Tomar las palabras que siguen al concepto
                following_words = context_words[concept_pos:concept_pos+8]
                answer = " ".join(following_words)
                
                # Limpiar la respuesta
                if len(answer) > 10 and len(answer) < 100:
                    return answer.capitalize()
            except:
                pass
        
        # Fallback: usar información del concepto
        if concept.get("related_terms"):
            return f"Un {concept['type']} relacionado con {', '.join(concept['related_terms'][:2])}"
        
        return f"Concepto central del texto sobre {concept_name}"
    
    def _generate_intelligent_distractors(
        self, main_concept: Dict, all_concepts: List[Dict], context: str
    ) -> List[str]:
        """Genera distractores inteligentes"""
        
        distractors = []
        
        # Usar otros conceptos como distractores
        for concept in all_concepts:
            if concept["name"] != main_concept["name"] and len(distractors) < 2:
                if concept.get("related_terms"):
                    distractor = f"Proceso relacionado con {concept['name']}"
                    distractors.append(distractor)
        
        # Completar con distractores genéricos pero creíbles
        generic_distractors = [
            "Método tradicional de análisis de datos",
            "Sistema convencional de procesamiento",
            "Técnica básica de información",
            "Herramienta estándar de clasificación",
            "Procedimiento manual de evaluación",
            "Algoritmo básico de comparación"
        ]
        
        while len(distractors) < 3:
            for distractor in generic_distractors:
                if distractor not in distractors and len(distractors) < 3:
                    distractors.append(distractor)
        
        return distractors
    
    def _generate_contextual_explanation(self, concept: Dict, correct_answer: str, context: str) -> str:
        """Genera una explicación contextual"""
        
        concept_name = concept["name"]
        
        if context:
            return f"La respuesta correcta se basa en la información del texto: '{context[:100]}...'. {concept_name} es un elemento clave mencionado en este contexto."
        
        return f"'{correct_answer}' es la respuesta correcta porque {concept_name} representa un concepto fundamental en el texto analizado y se relaciona directamente con los temas principales discutidos."
    
    def _create_enhanced_fallback_question(
        self, question_id: int, concepts: List[Dict], sentences: List[Dict], difficulty: str
    ) -> Dict:
        """Crea una pregunta de respaldo mejorada"""
        
        if concepts and len(concepts) > 0:
            concept_index = (question_id - 1) % len(concepts)
            main_concept = concepts[concept_index]
            concept_name = main_concept["name"]
        else:
            concept_name = "el tema principal"
        
        # Preguntas de respaldo más variadas
        fallback_questions = [
            f"¿Cuál es la característica más importante de {concept_name}?",
            f"Según el análisis del texto, ¿cómo se puede definir {concept_name}?",
            f"¿Qué función cumple {concept_name} en el contexto estudiado?",
            f"¿Por qué es relevante {concept_name} para el tema tratado?",
            f"¿Qué aspectos destacan de {concept_name} en el contenido analizado?"
        ]
        
        question_text = fallback_questions[(question_id - 1) % len(fallback_questions)]
        
        # Opciones más elaboradas
        if len(concepts) >= 4:
            options = [concept["name"] for concept in concepts[:4]]
            correct_answer = 0
        else:
            correct_option = concept_name if concept_name != "el tema principal" else "Concepto central del texto"
            options = [
                correct_option,
                "Método de análisis tradicional",
                "Sistema de procesamiento básico",
                "Herramienta de clasificación estándar"
            ]
            correct_answer = 0
        
        explanation = f"La respuesta correcta es '{options[correct_answer]}' ya que representa el concepto central identificado en el análisis del texto y se relaciona directamente con los temas principales discutidos."
        
        return {
            "id": question_id,
            "question": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": explanation,
            "difficulty": difficulty
        }
    
    async def generate_feedback(self, score: int, total: int, incorrect_questions: List[int], concepts: List[str]) -> str:
        """
        Genera retroalimentación pedagógica mejorada
        """
        try:
            percentage = (score / total) * 100
            
            # ✅ FEEDBACK ESTRUCTURADO Y PERSONALIZADO
            if percentage >= 80:
                base_feedback = f"¡Excelente trabajo! Has demostrado un sólido dominio de los conceptos clave"
                if concepts:
                    main_concepts = concepts[:2]
                    base_feedback += f" relacionados con **{' y '.join(main_concepts)}**"
                base_feedback += f". Tu puntuación de **{score}/{total} ({percentage:.1f}%)** indica una comprensión muy buena del tema."
                
                if concepts:
                    base_feedback += f"\n\n🎯 **Fortalezas identificadas:** Tienes un buen manejo de conceptos como {', '.join(concepts[:3])}."
                
            elif percentage >= 60:
                base_feedback = f"Buen trabajo. Has obtenido **{score} de {total}** respuestas correctas (**{percentage:.1f}%**)"
                if concepts:
                    base_feedback += f". Tienes una base sólida en **{concepts[0]}**"
                base_feedback += ", pero hay algunas áreas que puedes reforzar para mejorar tu comprensión."
                
                if len(concepts) > 1:
                    base_feedback += f"\n\n📚 **Áreas a reforzar:** Revisa especialmente los conceptos de {', '.join(concepts[1:3])}."
                
            else:
                base_feedback = f"Has obtenido **{score} de {total}** respuestas correctas (**{percentage:.1f}%**)"
                if concepts:
                    main_concepts = concepts[:2]
                    base_feedback += f". Te recomiendo revisar los conceptos fundamentales como **{' y '.join(main_concepts)}**"
                base_feedback += ". No te desanimes, el aprendizaje es un proceso gradual y cada intento te acerca más al dominio del tema."
                
                base_feedback += f"\n\n💪 **Plan de mejora:** Enfócate en comprender los conceptos básicos antes de avanzar a temas más complejos."
            
            return base_feedback
            
        except Exception as e:
            logger.error(f"Error generando feedback: {e}")
            return self._generate_fallback_feedback(score, total)
    
    def _generate_fallback_summary(self, text: str) -> str:
        """Genera un resumen básico sin IA"""
        sentences = text.split('.')[:3]
        key_terms = self._extract_key_terms(text)[:3]
        
        summary = f"📚 **Resumen básico:**\n\n"
        if key_terms:
            summary += f"🔑 **Conceptos principales:** {', '.join(key_terms)}\n\n"
        summary += '. '.join(sentences) + "."
        
        return summary
    
    def _generate_enhanced_fallback_quiz(self, text: str, concepts: List[str], num_questions: int) -> List[Dict]:
        """Genera preguntas básicas mejoradas"""
        questions = []
        processed_concepts = self._process_concepts_for_questions(concepts, text)
        
        for i in range(min(num_questions, max(3, len(processed_concepts)))):
            question = self._create_enhanced_fallback_question(
                i + 1, processed_concepts, [], "medium"
            )
            questions.append(question)
        return questions
    
    def _generate_fallback_feedback(self, score: int, total: int) -> str:
        """Genera feedback básico pero útil"""
        percentage = (score / total) * 100
        if percentage >= 80:
            return f"¡Excelente trabajo! Has obtenido **{score} de {total}** respuestas correctas (**{percentage:.1f}%**). Demuestras un sólido dominio del tema."
        elif percentage >= 60:
            return f"Buen trabajo. Has obtenido **{score} de {total}** respuestas correctas (**{percentage:.1f}%**). Tienes una base sólida, continúa practicando para mejorar."
        else:
            return f"Has obtenido **{score} de {total}** respuestas correctas (**{percentage:.1f}%**). Te recomiendo revisar el material de estudio y enfocarte en los conceptos principales. ¡El aprendizaje es un proceso, sigue adelante!"