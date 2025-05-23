# app/services/nlp_service.py
import spacy
import re
from collections import Counter
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class NLPService:
    def __init__(self):
        try:
            # Cargar modelo de spaCy en español
            self.nlp = spacy.load("es_core_news_sm")
        except OSError:
            # Si no está instalado, usar el modelo en inglés como fallback
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.nlp = None
                print("Warning: No spaCy model available. Install es_core_news_sm or en_core_web_sm")
    
    def extract_key_concepts(self, text: str, max_concepts: int = 10) -> List[Dict[str, Any]]:
        """
        Extrae conceptos clave del texto usando NLP
        """
        if not self.nlp:
            return self._fallback_key_concepts(text, max_concepts)
        
        doc = self.nlp(text)
        
        # Extraer entidades nombradas y sustantivos importantes
        concepts = []
        
        # Entidades nombradas
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                concepts.append(ent.text.lower().strip())
        
        # Sustantivos y frases nominales importantes
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Frases cortas
                concepts.append(chunk.text.lower().strip())
        
        # Contar frecuencias
        concept_freq = Counter(concepts)
        
        # Calcular relevancia usando TF-IDF
        if len(concepts) > 0:
            tfidf = TfidfVectorizer(max_features=max_concepts, stop_words='english')
            try:
                tfidf_matrix = tfidf.fit_transform([text])
                feature_names = tfidf.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]
                
                # Combinar frecuencia y relevancia TF-IDF
                result = []
                for concept, freq in concept_freq.most_common(max_concepts):
                    relevance = 0.5  # valor por defecto
                    if concept in feature_names:
                        idx = list(feature_names).index(concept)
                        relevance = scores[idx]
                    
                    result.append({
                        "concept": concept.title(),
                        "frequency": freq,
                        "relevance": float(relevance)
                    })
                
                return result[:max_concepts]
            except:
                # Fallback a solo frecuencia
                return [
                    {
                        "concept": concept.title(),
                        "frequency": freq,
                        "relevance": freq / len(concepts)
                    }
                    for concept, freq in concept_freq.most_common(max_concepts)
                ]
        
        return []
    
    def _fallback_key_concepts(self, text: str, max_concepts: int) -> List[Dict[str, Any]]:
        """
        Método de respaldo para extraer conceptos sin spaCy
        """
        # Limpiar y tokenizar
        words = re.findall(r'\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]{3,}\b', text.lower())
        
        # Palabras comunes a filtrar
        stop_words = {'que', 'por', 'para', 'con', 'una', 'como', 'más', 'pero', 'sus', 'les', 'muy'}
        
        # Filtrar palabras
        filtered_words = [word for word in words if word not in stop_words]
        
        # Contar frecuencias
        word_freq = Counter(filtered_words)
        
        return [
            {
                "concept": word.title(),
                "frequency": freq,
                "relevance": freq / len(filtered_words)
            }
            for word, freq in word_freq.most_common(max_concepts)
        ]
    
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """
        Analiza la complejidad del texto
        """
        sentences = re.split(r'[.!?]+', text)
        words = re.findall(r'\b\w+\b', text.lower())
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_words_per_sentence": len(words) / max(len(sentences), 1),
            "reading_time": max(1, len(words) // 200),  # ~200 palabras por minuto
            "complexity_level": self._determine_complexity(len(words), len(sentences))
        }
    
    def _determine_complexity(self, word_count: int, sentence_count: int) -> str:
        """
        Determina el nivel de complejidad del texto
        """
        avg_words = word_count / max(sentence_count, 1)
        
        if avg_words < 15 and word_count < 500:
            return "Básico"
        elif avg_words < 20 and word_count < 1500:
            return "Intermedio"
        else:
            return "Avanzado"
