# app/services/nlp_service.py - VERSIÓN MEJORADA UNIVERSAL
import spacy
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class UniversalNLPService:
    """
    Servicio NLP mejorado para cualquier tipo de texto en español
    """
    
    def __init__(self):
        self._load_spacy_model()
        self._setup_universal_patterns()
        self._setup_stop_words()
    
    def _load_spacy_model(self):
        """Carga el mejor modelo de spaCy disponible"""
        models_to_try = ["es_core_news_lg", "es_core_news_md", "es_core_news_sm"]
        
        self.nlp = None
        for model_name in models_to_try:
            try:
                self.nlp = spacy.load(model_name)
                logger.info(f"✅ Modelo spaCy cargado: {model_name}")
                break
            except OSError:
                continue
        
        if self.nlp is None:
            logger.warning("⚠️ No se pudo cargar modelo de spaCy en español")
            self._setup_fallback_processing()
    
    def _setup_fallback_processing(self):
        """Configura procesamiento sin spaCy"""
        logger.info("Configurando procesamiento básico sin spaCy")
        self.nlp = None
    
    def _setup_universal_patterns(self):
        """Configura patrones universales para diferentes dominios"""
        self.domain_patterns = {
            "historia": {
                "keywords": ["guerra", "batalla", "revolución", "imperio", "independencia", "tratado", 
                           "siglo", "época", "dinastía", "conquista", "invasión", "alianza"],
                "entities": ["país", "nación", "reino", "territorio", "ciudad", "región"],
                "temporal": r'\b(?:siglo\s+[IVX]+|(?:15|16|17|18|19|20)\d{2}|\d{1,2}\s+de\s+\w+\s+de\s+\d{4})\b',
                "people": r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+){1,2}\b'
            },
            "ciencia": {
                "keywords": ["proceso", "sistema", "organismo", "célula", "energía", "reacción",
                           "experimento", "investigación", "teoría", "hipótesis", "método", "análisis"],
                "entities": ["especie", "elemento", "compuesto", "proteína", "gen", "ecosistema"],
                "processes": r'\b\w*(?:ción|sis|oma|ema|génesis|lisis)\b',
                "scientific_terms": r'\b[A-Z][a-z]*(?:ina|asa|osa|ano|eno|ato|ido)\b'
            },
            "tecnologia": {
                "keywords": ["sistema", "software", "algoritmo", "datos", "aplicación", "programa",
                           "tecnología", "digital", "automático", "inteligencia", "red", "plataforma"],
                "entities": ["dispositivo", "servidor", "base", "interfaz", "protocolo", "framework"],
                "tech_terms": r'\b(?:API|CPU|GPU|RAM|HTTP|SQL|HTML|CSS|JavaScript|Python)\b',
                "concepts": r'\b(?:machine learning|deep learning|blockchain|cloud computing)\b'
            },
            "literatura": {
                "keywords": ["obra", "autor", "estilo", "narrativa", "poesía", "género", "movimiento",
                           "personaje", "argumento", "técnica", "estructura", "lenguaje"],
                "entities": ["novela", "cuento", "drama", "verso", "estrofa", "capítulo"],
                "literary_terms": r'\b(?:metáfora|símil|ironía|alegoría|realismo|modernismo)\b',
                "authors": r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+\b'
            },
            "economia": {
                "keywords": ["mercado", "precio", "demanda", "oferta", "inversión", "capital",
                           "empresa", "negocio", "financiero", "económico", "crecimiento", "desarrollo"],
                "entities": ["sector", "industria", "banco", "bolsa", "moneda", "producto"],
                "economic_terms": r'\b(?:PIB|GDP|inflación|deflación|recesión|devaluación)\b',
                "indicators": r'\b\d+[%]\b|\b\$\d+|\beuros?\b|\bdólares?\b'
            }
        }
    
    def _setup_stop_words(self):
        """Configura stop words completas en español"""
        self.stop_words = {
            # Artículos y determinantes
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'este', 'esta', 'estos', 'estas',
            'ese', 'esa', 'esos', 'esas', 'aquel', 'aquella', 'aquellos', 'aquellas',
            
            # Preposiciones
            'a', 'ante', 'bajo', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre', 'hacia',
            'hasta', 'mediante', 'para', 'por', 'según', 'sin', 'sobre', 'tras',
            
            # Conjunciones
            'y', 'e', 'o', 'u', 'pero', 'mas', 'sino', 'aunque', 'si', 'porque', 'como', 'cuando',
            'donde', 'mientras', 'que', 'quien', 'cual', 'cuyo',
            
            # Pronombres
            'yo', 'tú', 'él', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas', 'me', 'te', 'se',
            'nos', 'os', 'le', 'les', 'lo', 'la', 'los', 'las',
            
            # Verbos auxiliares y comunes
            'ser', 'estar', 'haber', 'tener', 'hacer', 'poder', 'deber', 'querer', 'saber', 'ver',
            'dar', 'ir', 'venir', 'decir', 'poner', 'salir', 'partir', 'llegar',
            
            # Adverbios comunes
            'muy', 'más', 'menos', 'mucho', 'poco', 'tanto', 'tan', 'también', 'tampoco', 'sí', 'no',
            'aquí', 'ahí', 'allí', 'cerca', 'lejos', 'arriba', 'abajo', 'dentro', 'fuera',
            'antes', 'después', 'ahora', 'luego', 'entonces', 'siempre', 'nunca', 'ya',
            
            # Otros
            'todo', 'cada', 'otro', 'mismo', 'algún', 'ningún', 'cualquier', 'tal', 'tanto',
            'cuanto', 'qué', 'cómo', 'cuándo', 'dónde', 'por qué'
        }
    
    def detect_text_domain(self, text: str) -> str:
        """Detecta el dominio del texto automáticamente"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = 0
            
            # Contar keywords del dominio
            for keyword in patterns["keywords"]:
                score += text_lower.count(keyword) * 2
            
            # Contar entidades del dominio
            for entity in patterns["entities"]:
                score += text_lower.count(entity)
            
            # Buscar patrones específicos
            if "temporal" in patterns:
                matches = len(re.findall(patterns["temporal"], text, re.IGNORECASE))
                score += matches * 3
            
            if "processes" in patterns:
                matches = len(re.findall(patterns["processes"], text, re.IGNORECASE))
                score += matches * 2
            
            domain_scores[domain] = score
        
        # Detectores específicos adicionales
        if any(word in text_lower for word in ["segunda guerra mundial", "primera guerra", "revolución francesa"]):
            domain_scores["historia"] = domain_scores.get("historia", 0) + 10
        
        if any(word in text_lower for word in ["fotosíntesis", "adn", "proteína", "evolución"]):
            domain_scores["ciencia"] = domain_scores.get("ciencia", 0) + 10
        
        if any(word in text_lower for word in ["inteligencia artificial", "algoritmo", "programación"]):
            domain_scores["tecnologia"] = domain_scores.get("tecnologia", 0) + 10
        
        # Retornar dominio con mayor puntuación
        if not domain_scores or max(domain_scores.values()) == 0:
            return "general"
        
        detected_domain = max(domain_scores, key=domain_scores.get)
        logger.info(f"🎯 Dominio detectado: {detected_domain} (score: {domain_scores[detected_domain]})")
        
        return detected_domain
    
    def extract_key_concepts(self, text: str, max_concepts: int = 10, domain: str = None) -> List[Dict[str, Any]]:
        """
        Extrae conceptos clave adaptándose al dominio del texto
        """
        # Detectar dominio si no se especifica
        if domain is None:
            domain = self.detect_text_domain(text)
        
        logger.info(f"🔍 Extrayendo conceptos para dominio: {domain}")
        
        # Usar spaCy si está disponible
        if self.nlp:
            return self._extract_concepts_with_spacy(text, max_concepts, domain)
        else:
            return self._extract_concepts_without_spacy(text, max_concepts, domain)
    
    def _extract_concepts_with_spacy(self, text: str, max_concepts: int, domain: str) -> List[Dict[str, Any]]:
        """Extrae conceptos usando spaCy"""
        doc = self.nlp(text)
        concepts = []
        
        # 1. Entidades nombradas importantes
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT', 'WORK_OF_ART', 'LAW']:
                if len(ent.text.strip()) > 2 and ent.text.lower() not in self.stop_words:
                    concepts.append({
                        "text": ent.text.strip(),
                        "type": "entity",
                        "label": ent.label_,
                        "frequency": text.lower().count(ent.text.lower()),
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
        
        # 2. Frases nominales relevantes
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if (self._is_valid_concept(chunk_text, domain) and 
                len(chunk_text.split()) <= 3 and 
                len(chunk_text) > 4):
                
                concepts.append({
                    "text": chunk_text,
                    "type": "noun_phrase", 
                    "frequency": text.lower().count(chunk_text.lower()),
                    "start": chunk.start_char,
                    "end": chunk.end_char
                })
        
        # 3. Términos técnicos del dominio
        domain_terms = self._extract_domain_specific_terms(text, domain)
        for term in domain_terms:
            concepts.append({
                "text": term,
                "type": "domain_term",
                "frequency": text.lower().count(term.lower()),
                "domain": domain
            })
        
        # 4. Sustantivos y adjetivos importantes
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                len(token.text) > 3 and 
                not token.is_stop and 
                not token.is_punct and 
                token.text.lower() not in self.stop_words and
                token.is_alpha):
                
                concepts.append({
                    "text": token.lemma_.title(),
                    "type": "important_term",
                    "pos": token.pos_,
                    "frequency": text.lower().count(token.text.lower())
                })
        
        return self._rank_and_filter_concepts(concepts, max_concepts, text, domain)
    
    def _extract_concepts_without_spacy(self, text: str, max_concepts: int, domain: str) -> List[Dict[str, Any]]:
        """Extrae conceptos sin spaCy usando procesamiento avanzado"""
        concepts = []
        
        # 1. Términos específicos del dominio
        domain_terms = self._extract_domain_specific_terms(text, domain)
        for term in domain_terms:
            concepts.append({
                "text": term,
                "type": "domain_term",
                "frequency": text.lower().count(term.lower()),
                "domain": domain
            })
        
        # 2. Nombres propios (capitalizados)
        proper_nouns = re.findall(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{2,}(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{2,}){0,2}\b', text)
        for noun in proper_nouns:
            if self._is_valid_concept(noun, domain):
                concepts.append({
                    "text": noun,
                    "type": "proper_noun",
                    "frequency": text.lower().count(noun.lower())
                })
        
        # 3. Palabras significativas (4+ caracteres, no stop words)
        words = re.findall(r'\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]{4,}\b', text)
        word_freq = Counter([word.lower() for word in words 
                           if word.lower() not in self.stop_words and self._is_valid_concept(word, domain)])
        
        for word, freq in word_freq.most_common(max_concepts * 2):
            concepts.append({
                "text": word.title(),
                "type": "significant_word",
                "frequency": freq
            })
        
        return self._rank_and_filter_concepts(concepts, max_concepts, text, domain)
    
    def _extract_domain_specific_terms(self, text: str, domain: str) -> List[str]:
        """Extrae términos específicos del dominio"""
        domain_terms = []
        
        if domain not in self.domain_patterns:
            return domain_terms
        
        patterns = self.domain_patterns[domain]
        
        # Buscar patrones específicos
        for pattern_name, pattern in patterns.items():
            if isinstance(pattern, str) and pattern.startswith(r'\b'):
                matches = re.findall(pattern, text, re.IGNORECASE)
                domain_terms.extend([match for match in matches if len(match) > 2])
        
        # Keywords del dominio que aparecen en el texto
        text_lower = text.lower()
        for keyword in patterns.get("keywords", []):
            if keyword in text_lower:
                domain_terms.append(keyword.title())
        
        return list(set(domain_terms))
    
    def _is_valid_concept(self, text: str, domain: str) -> bool:
        """Valida si un texto es un concepto válido"""
        text_clean = text.strip().lower()
        
        # Filtros básicos
        if (len(text_clean) < 3 or 
            text_clean in self.stop_words or
            text_clean.isdigit() or
            not re.match(r'^[a-záéíóúüñ\s]+$', text_clean)):
            return False
        
        # Filtros específicos por dominio
        invalid_patterns = {
            "historia": ["año", "años", "vez", "veces", "forma", "manera"],
            "ciencia": ["caso", "casos", "tipo", "tipos", "parte", "partes"],
            "tecnologia": ["usuario", "usuarios", "sistema", "datos"],
            "literatura": ["obra", "obras", "autor", "autores"],
            "economia": ["empresa", "empresas", "mercado", "sector"]
        }
        
        domain_invalid = invalid_patterns.get(domain, [])
        if text_clean in domain_invalid:
            return False
        
        return True
    
    def _rank_and_filter_concepts(self, concepts: List[Dict], max_concepts: int, 
                                text: str, domain: str) -> List[Dict[str, Any]]:
        """Rankea y filtra conceptos por relevancia"""
        
        # Consolidar conceptos duplicados
        concept_map = {}
        for concept in concepts:
            key = concept["text"].lower()
            if key in concept_map:
                # Mantener el de mayor frecuencia y mejor tipo
                existing = concept_map[key]
                if (concept["frequency"] > existing["frequency"] or
                    self._get_type_priority(concept["type"]) > self._get_type_priority(existing["type"])):
                    concept_map[key] = concept
            else:
                concept_map[key] = concept
        
        # Calcular relevancia
        unique_concepts = list(concept_map.values())
        for concept in unique_concepts:
            concept["relevance"] = self._calculate_relevance(concept, text, domain)
        
        # Ordenar por relevancia
        unique_concepts.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Formatear resultado final
        final_concepts = []
        for concept in unique_concepts[:max_concepts]:
            final_concepts.append({
                "concept": concept["text"],
                "frequency": concept["frequency"], 
                "relevance": round(concept["relevance"], 3),
                "type": concept.get("type", "general"),
                "domain": domain
            })
        
        return final_concepts
    
    def _get_type_priority(self, concept_type: str) -> int:
        """Asigna prioridad a tipos de conceptos"""
        priorities = {
            "entity": 5,
            "domain_term": 4,
            "proper_noun": 3,
            "noun_phrase": 2,
            "important_term": 1,
            "significant_word": 0
        }
        return priorities.get(concept_type, 0)
    
    def _calculate_relevance(self, concept: Dict, text: str, domain: str) -> float:
        """Calcula relevancia mejorada del concepto"""
        base_score = 0.1
        
        # Factor frecuencia normalizada
        max_freq = max([text.lower().count(word.lower()) for word in text.split()[:50]])
        freq_score = concept["frequency"] / max(max_freq, 1) * 0.3
        
        # Factor tipo de concepto
        type_score = self._get_type_priority(concept["type"]) * 0.15
        
        # Factor longitud (conceptos más largos suelen ser más específicos)
        length_score = min(len(concept["text"]) / 20, 0.2)
        
        # Factor posición (conceptos en título/inicio son más importantes)
        position_score = 0.2 if concept["text"].lower() in text[:200].lower() else 0.0
        
        # Factor dominio (conceptos del dominio detectado son más relevantes)
        domain_score = 0.2 if concept.get("domain") == domain or concept.get("type") == "domain_term" else 0.0
        
        # Factor capitalización (nombres propios)
        cap_score = 0.1 if concept["text"][0].isupper() else 0.0
        
        # Penalty por palabras muy comunes
        common_penalty = -0.1 if concept["text"].lower() in ["historia", "tiempo", "mundo", "parte"] else 0.0
        
        final_score = base_score + freq_score + type_score + length_score + position_score + domain_score + cap_score + common_penalty
        
        return max(0.0, min(1.0, final_score))
    
    def analyze_text_complexity(self, text: str, domain: str = None) -> Dict[str, Any]:
        """
        Analiza complejidad del texto adaptada al dominio
        """
        if domain is None:
            domain = self.detect_text_domain(text)
        
        # Métricas básicas
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = re.findall(r'\b\w+\b', text.lower())
        
        word_count = len(words)
        sentence_count = len(sentences)
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        
        # Métricas avanzadas
        unique_words = len(set(words))
        lexical_diversity = unique_words / max(word_count, 1)
        
        # Palabras técnicas del dominio
        domain_terms = self._extract_domain_specific_terms(text, domain)
        domain_term_ratio = len(domain_terms) / max(word_count, 1)
        
        # Palabras largas (7+ caracteres)
        long_words = [w for w in words if len(w) >= 7]
        long_word_ratio = len(long_words) / max(word_count, 1)
        
        # Determinar complejidad adaptada al dominio
        complexity_level = self._determine_domain_complexity(
            avg_words_per_sentence, lexical_diversity, long_word_ratio, 
            domain_term_ratio, domain, word_count
        )
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "unique_words": unique_words,
            "avg_words_per_sentence": round(avg_words_per_sentence, 1),
            "lexical_diversity": round(lexical_diversity, 3),
            "long_word_ratio": round(long_word_ratio, 3),
            "domain_term_ratio": round(domain_term_ratio, 3),
            "reading_time": max(1, word_count // 200),
            "complexity_level": complexity_level,
            "domain": domain,
            "estimated_education_level": self._estimate_education_level(complexity_level, domain)
        }
    
    def _determine_domain_complexity(self, avg_words: float, lexical_diversity: float,
                                   long_word_ratio: float, domain_term_ratio: float,
                                   domain: str, word_count: int) -> str:
        """Determina complejidad adaptada al dominio"""
        
        # Pesos específicos por dominio
        domain_weights = {
            "ciencia": {"technical": 2.0, "vocabulary": 1.5},
            "tecnologia": {"technical": 2.0, "vocabulary": 1.3},
            "literatura": {"technical": 1.2, "vocabulary": 2.0},
            "historia": {"technical": 1.5, "vocabulary": 1.5},
            "economia": {"technical": 1.8, "vocabulary": 1.4},
            "general": {"technical": 1.0, "vocabulary": 1.0}
        }
        
        weights = domain_weights.get(domain, domain_weights["general"])
        
        # Calcular puntuación de complejidad
        complexity_score = 0
        
        # Factor 1: Longitud de oraciones (ajustado por dominio)
        sentence_threshold = 20 if domain in ["ciencia", "tecnologia"] else 18
        if avg_words > sentence_threshold:
            complexity_score += 3
        elif avg_words > 12:
            complexity_score += 2
        else:
            complexity_score += 1
        
        # Factor 2: Diversidad léxica
        if lexical_diversity > 0.7:
            complexity_score += 3 * weights["vocabulary"]
        elif lexical_diversity > 0.5:
            complexity_score += 2 * weights["vocabulary"]
        else:
            complexity_score += 1
        
        # Factor 3: Términos técnicos del dominio
        if domain_term_ratio > 0.05:
            complexity_score += 3 * weights["technical"]
        elif domain_term_ratio > 0.02:
            complexity_score += 2 * weights["technical"]
        else:
            complexity_score += 1
        
        # Factor 4: Palabras largas
        if long_word_ratio > 0.25:
            complexity_score += 3
        elif long_word_ratio > 0.15:
            complexity_score += 2
        else:
            complexity_score += 1
        
        # Factor 5: Longitud total
        if word_count > 1000:
            complexity_score += 1
        
        # Clasificación final
        if complexity_score >= 10:
            return "Avanzado"
        elif complexity_score >= 7:
            return "Intermedio"
        else:
            return "Básico"
    
    def _estimate_education_level(self, complexity: str, domain: str) -> str:
        """Estima nivel educativo requerido"""
        
        level_mapping = {
            "Básico": {
                "general": "Secundaria",
                "historia": "Secundaria", 
                "literatura": "Secundaria",
                "ciencia": "Bachillerato",
                "tecnologia": "Bachillerato",
                "economia": "Bachillerato"
            },
            "Intermedio": {
                "general": "Bachillerato",
                "historia": "Bachillerato",
                "literatura": "Universitario",
                "ciencia": "Universitario", 
                "tecnologia": "Universitario",
                "economia": "Universitario"
            },
            "Avanzado": {
                "general": "Universitario",
                "historia": "Universitario",
                "literatura": "Posgrado",
                "ciencia": "Posgrado",
                "tecnologia": "Especialización",
                "economia": "Posgrado"
            }
        }
        
        return level_mapping.get(complexity, {}).get(domain, "Universitario")
    
    def extract_key_sentences(self, text: str, max_sentences: int = 5) -> List[Dict[str, Any]]:
        """Extrae las oraciones más importantes del texto"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
        
        if len(sentences) <= max_sentences:
            return [{"text": s, "importance": 1.0, "position": i} for i, s in enumerate(sentences)]
        
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            score = self._calculate_sentence_importance(sentence, text, i, len(sentences))
            sentence_scores.append({
                "text": sentence,
                "importance": score,
                "position": i,
                "length": len(sentence.split())
            })
        
        # Ordenar por importancia
        sentence_scores.sort(key=lambda x: x["importance"], reverse=True)
        
        return sentence_scores[:max_sentences]
    
    def _calculate_sentence_importance(self, sentence: str, full_text: str, 
                                     position: int, total_sentences: int) -> float:
        """Calcula la importancia de una oración"""
        score = 0.0
        
        # Factor posición (primera y última son más importantes)
        if position == 0:
            score += 0.3
        elif position == total_sentences - 1:
            score += 0.2
        elif position < total_sentences * 0.2:  # Primer 20%
            score += 0.15
        
        # Factor longitud (oraciones ni muy cortas ni muy largas)
        word_count = len(sentence.split())
        if 10 <= word_count <= 25:
            score += 0.2
        elif word_count > 30:
            score -= 0.1
        
        # Factor términos clave (cuántas palabras importantes contiene)
        important_words = re.findall(r'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{4,}\b', sentence)
        score += min(len(important_words) * 0.05, 0.3)
        
        # Factor conectores (oraciones con "por tanto", "además", etc.)
        connectors = ["por tanto", "además", "sin embargo", "en consecuencia", "por lo tanto", 
                     "finalmente", "en resumen", "principalmente", "especialmente"]
        if any(conn in sentence.lower() for conn in connectors):
            score += 0.15
        
        return min(score, 1.0)
    
    def extract_temporal_information(self, text: str) -> Dict[str, List[str]]:
        """Extrae información temporal del texto"""
        temporal_info = {
            "dates": [],
            "periods": [],
            "temporal_expressions": []
        }
        
        # Fechas específicas
        date_patterns = [
            r'\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b',  # 12 de octubre de 1492
            r'\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}\b',
            r'\b\d{4}\b',  # Años
            r'\b(?:19|20)\d{2}\b'  # Siglos XX y XXI
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            temporal_info["dates"].extend(matches)
        
        # Períodos históricos
        period_patterns = [
            r'\bsiglo\s+[IVX]+\b',
            r'\b(?:primera|segunda)\s+guerra\s+mundial\b',
            r'\brevolución\s+\w+\b',
            r'\bedad\s+(?:media|moderna|contemporánea)\b',
            r'\brenacimiento\b',
            r'\bilustración\b'
        ]
        
        for pattern in period_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            temporal_info["periods"].extend(matches)
        
        # Expresiones temporales
        temporal_expressions = [
            r'\b(?:durante|después de|antes de|a partir de|hasta)\s+\w+',
            r'\ben\s+(?:el año|la época|el período)\s+\w+',
            r'\b(?:inicialmente|posteriormente|finalmente|actualmente)\b'
        ]
        
        for pattern in temporal_expressions:
            matches = re.findall(pattern, text, re.IGNORECASE)
            temporal_info["temporal_expressions"].extend(matches)
        
        # Eliminar duplicados y limpiar
        for key in temporal_info:
            temporal_info[key] = list(set(temporal_info[key]))
        
        return temporal_info
    
    def get_text_structure(self, text: str) -> Dict[str, Any]:
        """Analiza la estructura del texto"""
        structure = {
            "paragraphs": [],
            "sections": [],
            "has_titles": False,
            "has_lists": False,
            "total_paragraphs": 0
        }
        
        # Dividir en párrafos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        structure["total_paragraphs"] = len(paragraphs)
        
        for i, paragraph in enumerate(paragraphs):
            para_info = {
                "index": i,
                "length": len(paragraph),
                "word_count": len(paragraph.split()),
                "is_title": self._is_title_paragraph(paragraph),
                "is_list": self._is_list_paragraph(paragraph)
            }
            structure["paragraphs"].append(para_info)
            
            if para_info["is_title"]:
                structure["has_titles"] = True
            if para_info["is_list"]:
                structure["has_lists"] = True
        
        return structure
    
    def _is_title_paragraph(self, paragraph: str) -> bool:
        """Determina si un párrafo es un título"""
        return (len(paragraph) < 100 and 
                paragraph.isupper() or 
                (paragraph[0].isupper() and paragraph.count('.') == 0))
    
    def _is_list_paragraph(self, paragraph: str) -> bool:
        """Determina si un párrafo contiene una lista"""
        list_indicators = [r'^\s*\d+\.', r'^\s*-', r'^\s*\*', r'^\s*•']
        return any(re.search(pattern, paragraph, re.MULTILINE) for pattern in list_indicators)


# Alias para compatibilidad con el código existente
NLPService = UniversalNLPService