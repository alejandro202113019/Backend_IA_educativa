# app/services/nlp_service.py - VERSIÓN MEJORADA CRÍTICA
import spacy
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class NLPService:
    def __init__(self):
        try:
            # Cargar modelo de spaCy en español (más grande y preciso)
            self.nlp = spacy.load("es_core_news_md")  # ← CAMBIO: usar modelo mediano
        except OSError:
            try:
                # Fallback al modelo pequeño
                self.nlp = spacy.load("es_core_news_sm")
            except OSError:
                try:
                    # Último recurso: modelo en inglés
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    self.nlp = None
                    print("Warning: No spaCy model available")
        
        # Stop words expandidas y mejoradas
        self.stop_words_spanish = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 
            'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 
            'las', 'una', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 
            'si', 'está', 'han', 'ser', 'muy', 'puede', 'tiene', 'entre', 'todo',
            'también', 'cuando', 'donde', 'cual', 'cada', 'desde', 'sobre', 'hasta',
            'fue', 'era', 'eran', 'fueron', 'sido', 'siendo', 'había', 'habían',
            'esto', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
            'aquel', 'aquella', 'aquellos', 'aquellas', 'año', 'años', 'parte',
            'través', 'durante', 'después', 'antes', 'mientras', 'aunque', 'porque'
        }
    
    def extract_key_concepts(self, text: str, max_concepts: int = 10) -> List[Dict[str, Any]]:
        """
        Extrae conceptos clave del texto usando NLP MEJORADO
        """
        if not self.nlp:
            return self._fallback_key_concepts(text, max_concepts)
        
        # 🔥 MEJORADO: Preprocesar texto más inteligentemente
        clean_text = self._intelligent_text_cleaning(text)
        doc = self.nlp(clean_text)
        
        # 🔥 NUEVO: Estrategia multi-nivel para extraer conceptos
        concepts = []
        
        # 1. ENTIDADES NOMBRADAS (más específicas)
        named_entities = self._extract_named_entities(doc)
        concepts.extend(named_entities)
        
        # 2. FRASES NOMINALES IMPORTANTES
        noun_phrases = self._extract_meaningful_noun_phrases(doc)
        concepts.extend(noun_phrases)
        
        # 3. SUSTANTIVOS Y ADJETIVOS RELEVANTES
        significant_tokens = self._extract_significant_tokens(doc)
        concepts.extend(significant_tokens)
        
        # 🔥 MEJORADO: Filtrar y consolidar conceptos
        filtered_concepts = self._filter_and_consolidate_concepts(concepts, text)
        
        # 🔥 NUEVO: Calcular relevancia mejorada con TF-IDF
        final_concepts = self._calculate_enhanced_relevance(filtered_concepts, text, max_concepts)
        
        return final_concepts[:max_concepts]
    
    def _intelligent_text_cleaning(self, text: str) -> str:
        """Limpieza inteligente que preserva información importante"""
        # Remover metadatos de PDF
        text = re.sub(r'--- Página \d+ ---', '', text)
        text = re.sub(r'IES [^–]+–[^A-Z]*', '', text)
        text = re.sub(r'Departamento de [^A-Z]*', '', text)
        
        # Normalizar espacios pero preservar estructura
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def _extract_named_entities(self, doc) -> List[str]:
        """Extrae entidades nombradas relevantes"""
        concepts = []
        
        for ent in doc.ents:
            # Filtrar entidades relevantes por tipo
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'FAC', 'PRODUCT', 'WORK_OF_ART']:
                entity_text = ent.text.strip()
                # Filtrar entidades muy cortas o genéricas
                if len(entity_text) > 3 and entity_text.lower() not in self.stop_words_spanish:
                    concepts.append(entity_text)
        
        return concepts
    
    def _extract_meaningful_noun_phrases(self, doc) -> List[str]:
        """Extrae frases nominales significativas"""
        concepts = []
        
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            
            # Filtros mejorados para frases nominales
            if (2 <= len(chunk_text.split()) <= 4 and  # Longitud razonable
                len(chunk_text) > 6 and  # No muy corto
                chunk_text.lower() not in self.stop_words_spanish and
                not re.match(r'^(la|el|un|una)\s+\w+$', chunk_text.lower())):  # No solo artículo + palabra
                
                concepts.append(chunk_text)
        
        return concepts
    
    def _extract_significant_tokens(self, doc) -> List[str]:
        """Extrae tokens individuales significativos"""
        concepts = []
        
        for token in doc:
            # Criterios mejorados para tokens significativos
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and  # Solo sustantivos, nombres propios, adjetivos
                len(token.text) > 4 and  # Mínimo 5 caracteres
                token.text.lower() not in self.stop_words_spanish and
                not token.is_punct and
                not token.is_space and
                not token.like_num and
                token.is_alpha):  # Solo letras
                
                concepts.append(token.lemma_.title())
        
        return concepts
    
    def _filter_and_consolidate_concepts(self, concepts: List[str], original_text: str) -> List[str]:
        """Filtra conceptos duplicados y los consolida inteligentemente"""
        # Contar frecuencias
        concept_freq = Counter([concept.lower() for concept in concepts])
        
        # Filtrar conceptos por frecuencia mínima
        min_frequency = 1 if len(original_text) < 1000 else 2
        
        filtered = []
        for concept in concepts:
            if (concept_freq[concept.lower()] >= min_frequency and
                concept not in filtered):  # Evitar duplicados exactos
                filtered.append(concept)
        
        return filtered
    
    def _calculate_enhanced_relevance(self, concepts: List[str], text: str, max_concepts: int) -> List[Dict[str, Any]]:
        """Calcula relevancia mejorada combinando múltiples factores"""
        if not concepts:
            return []
        
        # Contar frecuencias
        concept_freq = Counter([concept.lower() for concept in concepts])
        
        # Calcular TF-IDF si es posible
        tfidf_scores = {}
        try:
            # Crear corpus con conceptos únicos
            unique_concepts = list(set([concept.lower() for concept in concepts]))
            if len(unique_concepts) > 1:
                vectorizer = TfidfVectorizer(vocabulary=unique_concepts, lowercase=True)
                tfidf_matrix = vectorizer.fit_transform([text.lower()])
                feature_names = vectorizer.get_feature_names_out()
                
                for i, concept in enumerate(feature_names):
                    tfidf_scores[concept] = tfidf_matrix[0, i]
        except Exception as e:
            print(f"Warning: TF-IDF calculation failed: {e}")
        
        # Crear resultados finales
        results = []
        processed_concepts = set()
        
        for concept in concepts:
            concept_lower = concept.lower()
            
            if concept_lower not in processed_concepts:
                frequency = concept_freq[concept_lower]
                tfidf_score = tfidf_scores.get(concept_lower, 0.1)
                
                # 🔥 NUEVO: Scoring mejorado
                relevance_score = self._calculate_concept_relevance(
                    concept, frequency, tfidf_score, text
                )
                
                results.append({
                    "concept": concept.title(),
                    "frequency": frequency,
                    "relevance": float(relevance_score)
                })
                
                processed_concepts.add(concept_lower)
        
        # Ordenar por relevancia combinada
        results.sort(key=lambda x: (x["relevance"] * 0.7 + (x["frequency"] / len(concepts)) * 0.3), reverse=True)
        
        return results
    
    def _calculate_concept_relevance(self, concept: str, frequency: int, tfidf_score: float, text: str) -> float:
        """Calcula un score de relevancia más sofisticado"""
        base_score = tfidf_score if tfidf_score > 0 else 0.1
        
        # Bonus por longitud (conceptos más largos suelen ser más específicos)
        length_bonus = min(len(concept) / 20, 0.3)
        
        # Bonus por posición (conceptos en el título o inicio son más importantes)
        position_bonus = 0.2 if concept.lower() in text[:200].lower() else 0.0
        
        # Bonus por capitalización (nombres propios)
        capitalization_bonus = 0.1 if concept[0].isupper() else 0.0
        
        # Penalty por palabras muy comunes en español
        common_words = {'guerra', 'mundo', 'historia', 'tiempo', 'país', 'países'}
        common_penalty = -0.1 if concept.lower() in common_words else 0.0
        
        final_score = base_score + length_bonus + position_bonus + capitalization_bonus + common_penalty
        
        return max(0.0, min(1.0, final_score))  # Clamp entre 0 y 1
    
    def _fallback_key_concepts(self, text: str, max_concepts: int) -> List[Dict[str, Any]]:
        """
        Método de respaldo MEJORADO para extraer conceptos sin spaCy
        """
        # Limpieza inteligente
        clean_text = self._intelligent_text_cleaning(text)
        
        # Tokenización mejorada con regex
        words = re.findall(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{3,}\b', clean_text)  # Solo palabras capitalizadas de 4+ letras
        
        # Filtrar palabras
        filtered_words = [
            word for word in words 
            if word.lower() not in self.stop_words_spanish and len(word) > 4
        ]
        
        # Contar frecuencias
        word_freq = Counter(filtered_words)
        
        # Crear resultados con relevancia básica
        results = []
        for word, freq in word_freq.most_common(max_concepts):
            relevance = freq / len(filtered_words) if filtered_words else 0.1
            results.append({
                "concept": word,
                "frequency": freq,
                "relevance": float(relevance)
            })
        
        return results
    
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """
        Analiza la complejidad del texto con métricas mejoradas
        """
        # Limpieza básica
        clean_text = self._intelligent_text_cleaning(text)
        
        # Contar elementos básicos
        sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if s.strip()]
        words = re.findall(r'\b\w+\b', clean_text.lower())
        
        # Métricas básicas
        word_count = len(words)
        sentence_count = len(sentences)
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        
        # 🔥 NUEVO: Métricas avanzadas
        unique_words = len(set(words))
        lexical_diversity = unique_words / max(word_count, 1)
        
        # Palabras largas (indicador de complejidad)
        long_words = [w for w in words if len(w) > 6]
        long_word_ratio = len(long_words) / max(word_count, 1)
        
        # Determinar complejidad mejorada
        complexity_level = self._determine_enhanced_complexity(
            avg_words_per_sentence, lexical_diversity, long_word_ratio, word_count
        )
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "unique_words": unique_words,
            "avg_words_per_sentence": round(avg_words_per_sentence, 1),
            "lexical_diversity": round(lexical_diversity, 2),
            "long_word_ratio": round(long_word_ratio, 2),
            "reading_time": max(1, word_count // 200),  # ~200 palabras por minuto
            "complexity_level": complexity_level
        }
    
    def _determine_enhanced_complexity(self, avg_words: float, lexical_diversity: float, 
                                     long_word_ratio: float, word_count: int) -> str:
        """
        Determina el nivel de complejidad con criterios mejorados
        """
        # Sistema de puntuación
        complexity_score = 0
        
        # Factor 1: Longitud promedio de oraciones
        if avg_words > 25:
            complexity_score += 3
        elif avg_words > 15:
            complexity_score += 2
        else:
            complexity_score += 1
        
        # Factor 2: Diversidad léxica
        if lexical_diversity > 0.7:
            complexity_score += 3
        elif lexical_diversity > 0.5:
            complexity_score += 2
        else:
            complexity_score += 1
        
        # Factor 3: Proporción de palabras largas
        if long_word_ratio > 0.2:
            complexity_score += 3
        elif long_word_ratio > 0.1:
            complexity_score += 2
        else:
            complexity_score += 1
        
        # Factor 4: Longitud total del texto
        if word_count > 2000:
            complexity_score += 1
        
        # Clasificación final
        if complexity_score >= 8:
            return "Avanzado"
        elif complexity_score >= 6:
            return "Intermedio"
        else:
            return "Básico"