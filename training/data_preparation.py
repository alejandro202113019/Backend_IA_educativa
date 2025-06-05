# training/data_preparation.py - CORREGIDO SIN EMOJIS
import random
import re
import logging
from typing import List, Dict, Any
from datasets import load_dataset
from collections import Counter

logger = logging.getLogger(__name__)

class EducationalDataGenerator:
    """
    Generador de datos sintéticos para entrenamiento educativo
    """
    
    def __init__(self):
        logger.info("Inicializando generador de datos educativos")
        self._load_base_datasets()
    
    def _load_base_datasets(self):
        """Carga datasets públicos como base"""
        try:
            logger.info("Cargando datasets públicos...")
            
            # Dataset para resúmenes
            self.cnn_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1000]")
            
            # Dataset para Q&A
            self.squad_dataset = load_dataset("squad_v2", split="train[:1000]")
            
            logger.info("Datasets públicos cargados")
            
        except Exception as e:
            logger.warning(f"Error cargando datasets públicos: {e}")
            self.cnn_dataset = None
            self.squad_dataset = None
    
    def generate_summary_data(self, n_samples: int = 500) -> List[Dict[str, Any]]:
        """Genera datos para fine-tuning de resúmenes educativos"""
        logger.info(f"Generando {n_samples} ejemplos de resúmenes educativos")
        
        educational_summaries = []
        
        # Usar CNN/DailyMail como base si está disponible
        if self.cnn_dataset:
            for i, item in enumerate(self.cnn_dataset):
                if len(educational_summaries) >= n_samples:
                    break
                
                try:
                    # Transformar a formato educativo
                    educational_summary = self._create_educational_summary(
                        item['article'], item['highlights']
                    )
                    educational_summaries.append(educational_summary)
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Procesados {i + 1} artículos")
                
                except Exception as e:
                    logger.warning(f"Error procesando artículo {i}: {e}")
                    continue
        
        # Generar ejemplos sintéticos adicionales si es necesario
        while len(educational_summaries) < n_samples:
            synthetic_example = self._create_synthetic_summary_example()
            educational_summaries.append(synthetic_example)
        
        logger.info(f"Generados {len(educational_summaries)} ejemplos de resúmenes")
        return educational_summaries[:n_samples]
    
    def _create_educational_summary(self, article: str, highlights: str) -> Dict[str, Any]:
        """Convierte un artículo en un ejemplo de resumen educativo"""
        
        # Extraer conceptos clave del artículo
        key_concepts = self._extract_key_concepts_simple(article)
        
        # Crear resumen educativo estructurado
        educational_summary = f"""RESUMEN EDUCATIVO:

CONCEPTOS CLAVE: {', '.join(key_concepts[:3])}

CONTENIDO PRINCIPAL:
{highlights}

PARA RECORDAR: Los puntos más importantes de este contenido son {', '.join(key_concepts[:2])}, que se relacionan directamente con el tema central del texto."""
        
        return {
            "input_text": article[:1000],  # Limitar longitud
            "target_text": educational_summary,
            "concepts": key_concepts,
            "educational_level": random.choice(["básico", "intermedio", "avanzado"])
        }
    
    def _create_synthetic_summary_example(self) -> Dict[str, Any]:
        """Crea ejemplo sintético de resumen"""
        topics = [
            "inteligencia artificial", "cambio climático", "historia mundial",
            "biología celular", "física cuántica", "literatura clásica",
            "economía global", "tecnología blockchain", "psicología cognitiva"
        ]
        
        topic = random.choice(topics)
        
        return {
            "input_text": f"Este es un texto educativo sobre {topic}. El tema abarca múltiples aspectos importantes que los estudiantes deben comprender para desarrollar una comprensión integral del tema. Los conceptos fundamentales incluyen definiciones básicas, aplicaciones prácticas y implicaciones futuras.",
            "target_text": f"""RESUMEN EDUCATIVO sobre {topic.title()}:

CONCEPTOS CLAVE: {topic}, aplicaciones, fundamentos

PUNTOS PRINCIPALES: El texto explora los aspectos fundamentales de {topic}, incluyendo sus principales características y aplicaciones en el mundo actual.

PARA RECORDAR: {topic.title()} es un campo de estudio esencial que requiere comprensión de conceptos básicos y aplicaciones prácticas.""",
            "concepts": [topic, "fundamentos", "aplicaciones"],
            "educational_level": "intermedio"
        }
    
    def generate_question_data(self, n_samples: int = 500) -> List[Dict[str, Any]]:
        """Genera datos para fine-tuning de generación de preguntas"""
        logger.info(f"Generando {n_samples} ejemplos de preguntas educativas")
        
        qa_examples = []
        
        # Usar SQuAD como base si está disponible
        if self.squad_dataset:
            for i, item in enumerate(self.squad_dataset):
                if len(qa_examples) >= n_samples:
                    break
                
                try:
                    qa_example = self._create_educational_qa(item)
                    qa_examples.append(qa_example)
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Procesadas {i + 1} preguntas")
                
                except Exception as e:
                    logger.warning(f"Error procesando pregunta {i}: {e}")
                    continue
        
        # Generar ejemplos sintéticos adicionales
        while len(qa_examples) < n_samples:
            synthetic_qa = self._create_synthetic_qa_example()
            qa_examples.append(synthetic_qa)
        
        logger.info(f"Generados {len(qa_examples)} ejemplos de Q&A")
        return qa_examples[:n_samples]
    
    def _create_educational_qa(self, squad_item: Dict) -> Dict[str, Any]:
        """Convierte un item de SQuAD en ejemplo educativo"""
        
        context = squad_item['context']
        original_question = squad_item['question']
        
        # Mejorar la pregunta para ser más educativa
        educational_question = self._improve_question_educational(original_question)
        
        # Crear prompt para generación
        input_prompt = f"Contexto: {context}\n\nGenera una pregunta educativa sobre este contenido:"
        
        return {
            "input_text": input_prompt,
            "target_text": educational_question,
            "context": context,
            "original_question": original_question,
            "difficulty": random.choice(["fácil", "medio", "difícil"])
        }
    
    def _improve_question_educational(self, question: str) -> str:
        """Mejora una pregunta para hacerla más educativa"""
        
        educational_starters = [
            "¿Cómo se puede explicar",
            "¿Cuál es la importancia de",
            "¿Qué factores influyen en",
            "¿Por qué es relevante",
            "¿De qué manera se relaciona"
        ]
        
        # 40% de probabilidad de usar un starter educativo
        if random.random() < 0.4:
            starter = random.choice(educational_starters)
            # Adaptar la pregunta original
            question_lower = question.lower()
            if question_lower.startswith("what"):
                return f"{starter} {question[4:].lower()}?"
            elif question_lower.startswith("who"):
                return f"{starter} {question[3:].lower()}?"
            elif question_lower.startswith("when"):
                return f"{starter} {question[4:].lower()}?"
            else:
                return f"{starter} que {question.lower()}"
        
        return question
    
    def _create_synthetic_qa_example(self) -> Dict[str, Any]:
        """Crea ejemplo sintético de Q&A"""
        
        contexts = [
            "La fotosíntesis es el proceso mediante el cual las plantas convierten la luz solar en energía química. Este proceso es fundamental para la vida en la Tierra ya que produce oxígeno y glucosa.",
            "La revolución industrial marcó un punto de inflexión en la historia humana, transformando la manera en que trabajamos y vivimos. Introdujo nuevas tecnologías y cambió la estructura social.",
            "Los algoritmos de aprendizaje automático permiten a las computadoras aprender patrones de los datos sin ser explícitamente programadas para cada tarea específica.",
        ]
        
        questions = [
            "¿Cuál es la importancia de la fotosíntesis en los ecosistemas terrestres?",
            "¿Cómo transformó la revolución industrial la estructura social moderna?",
            "¿Qué ventajas ofrecen los algoritmos de machine learning en el procesamiento de datos?"
        ]
        
        idx = random.randint(0, len(contexts) - 1)
        
        return {
            "input_text": f"Contexto: {contexts[idx]}\n\nGenera una pregunta educativa sobre este contenido:",
            "target_text": questions[idx],
            "context": contexts[idx],
            "difficulty": "medio"
        }
    
    def generate_feedback_data(self, n_samples: int = 300) -> List[Dict[str, Any]]:
        """Genera datos para fine-tuning de feedback educativo"""
        logger.info(f"Generando {n_samples} ejemplos de feedback educativo")
        
        feedback_examples = []
        
        for i in range(n_samples):
            feedback_example = self._create_feedback_example()
            feedback_examples.append(feedback_example)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Generados {i + 1} ejemplos de feedback")
        
        logger.info(f"Generados {len(feedback_examples)} ejemplos de feedback")
        return feedback_examples
    
    def _create_feedback_example(self) -> Dict[str, Any]:
        """Crea ejemplo de feedback personalizado"""
        
        # Simular resultados de quiz
        total_questions = random.randint(5, 15)
        score = random.randint(0, total_questions)
        percentage = (score / total_questions) * 100
        
        concepts = random.sample([
            "inteligencia artificial", "machine learning", "algoritmos",
            "datos", "análisis", "programación", "estadística"
        ], k=random.randint(2, 4))
        
        incorrect_questions = random.sample(
            range(1, total_questions + 1), 
            k=max(0, total_questions - score)
        )
        
        # Crear input para el modelo
        input_text = f"""Resultados del quiz:
Puntuación: {score}/{total_questions} ({percentage:.1f}%)
Conceptos evaluados: {', '.join(concepts)}
Preguntas incorrectas: {incorrect_questions if incorrect_questions else 'Ninguna'}

Genera feedback personalizado y constructivo:"""
        
        # Crear feedback objetivo
        target_feedback = self._generate_structured_feedback(
            score, total_questions, percentage, concepts, incorrect_questions
        )
        
        return {
            "input_text": input_text,
            "target_text": target_feedback,
            "score": score,
            "total": total_questions,
            "concepts": concepts
        }
    
    def _generate_structured_feedback(self, score: int, total: int, percentage: float, 
                                    concepts: List[str], incorrect_questions: List[int]) -> str:
        """Genera feedback estructurado"""
        
        if percentage >= 85:
            feedback = f"""EXCELENTE TRABAJO! Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%).

FORTALEZAS IDENTIFICADAS:
- Dominas muy bien los conceptos de {concepts[0]} y {concepts[1] if len(concepts) > 1 else 'los temas principales'}
- Tu comprensión del material es sólida y consistente

PRÓXIMOS PASOS:
- Explora temas más avanzados relacionados con {concepts[0]}
- Comparte tu conocimiento ayudando a otros estudiantes"""

        elif percentage >= 70:
            feedback = f"""BUEN TRABAJO. Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%).

FORTALEZAS IDENTIFICADAS:
- Tienes una base sólida en {concepts[0]}
- Comprendes bien los conceptos fundamentales

ÁREAS DE MEJORA:
- Repasa los conceptos de {concepts[-1]} donde tuviste algunas dificultades
- Practica más ejercicios similares para consolidar el aprendizaje"""

        else:
            feedback = f"""SIGUE ADELANTE! Has obtenido {score} de {total} respuestas correctas ({percentage:.1f}%).

PLAN DE MEJORA:
- Enfócate en revisar los conceptos básicos de {concepts[0]} y {concepts[1] if len(concepts) > 1 else 'el tema principal'}
- Toma tiempo para entender cada concepto antes de avanzar
- Practica con ejemplos simples antes de intentar problemas complejos

RECUERDA: El aprendizaje es un proceso gradual. Cada intento te acerca más al dominio del tema."""
        
        return feedback
    
    def _extract_key_concepts_simple(self, text: str) -> List[str]:
        """Extracción simple de conceptos clave"""
        # Limpiar texto
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()
        
        # Filtrar stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'el', 'la', 'los', 'las',
            'un', 'una', 'de', 'del', 'que', 'para', 'con', 'por', 'en', 'es', 'se'
        }
        
        # Palabras significativas (mínimo 4 caracteres)
        significant_words = [
            word for word in words 
            if len(word) >= 4 and word not in stop_words
        ]
        
        # Contar frecuencias y devolver las más comunes
        word_freq = Counter(significant_words)
        return [word.title() for word, _ in word_freq.most_common(8)]