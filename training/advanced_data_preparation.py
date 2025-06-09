# training/advanced_data_preparation.py - GENERADOR AVANZADO DE DATOS EDUCATIVOS
import random
import re
import json
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter
import os

logger = logging.getLogger(__name__)

class AdvancedEducationalDataGenerator:
    """
    Generador avanzado de datos sintéticos para fine-tuning educativo
    """
    
    def __init__(self):
        logger.info("🚀 Inicializando generador avanzado de datos educativos")
        self._setup_comprehensive_templates()
        self._setup_question_generation_patterns()
        self._setup_feedback_patterns()
    
    def _setup_comprehensive_templates(self):
        """Configura plantillas comprehensivas por dominio"""
        self.domain_templates = {
            "historia_segunda_guerra_mundial": {
                "texts": [
                    """La Segunda Guerra Mundial (1939-1945) fue el conflicto más devastador de la historia humana. Comenzó cuando Alemania Nazi, liderada por Adolf Hitler, invadió Polonia el 1 de septiembre de 1939. Este evento provocó que Francia y Reino Unido declararan la guerra a Alemania, iniciando así una guerra global que eventualmente involucraría a la mayoría de las naciones del mundo.

El conflicto se caracterizó por innovaciones tecnológicas militares sin precedentes, incluyendo tanques avanzados, aviación de combate, radar, y eventualmente armas nucleares. La guerra se libró en múltiples frentes: el Frente Occidental en Europa, el Frente Oriental contra la Unión Soviética, el Teatro del Pacífico contra Japón, y campañas en África del Norte y el Mediterráneo.

Los principales bandos fueron las Potencias del Eje (Alemania, Italia, Japón) contra los Aliados (Reino Unido, Unión Soviética, Estados Unidos, China y otros). La guerra pasó por varias fases: victorias iniciales del Eje (1939-1942), el punto de inflexión (1942-1943), y la contraofensiva aliada (1943-1945).

Eventos clave incluyeron la Batalla de Francia (1940), la Batalla de Inglaterra, la Operación Barbarroja (invasión alemana de la URSS), el ataque a Pearl Harbor que llevó a Estados Unidos a la guerra, las batallas de Stalingrado y Midway que marcaron el cambio de rumbo, y el Desembarco de Normandía (Día D) que abrió el segundo frente en Europa.

La guerra terminó con la rendición incondicional de Alemania el 8 de mayo de 1945, seguida por la rendición de Japón el 2 de septiembre de 1945, después de que Estados Unidos lanzara bombas atómicas sobre Hiroshima y Nagasaki.

Las consecuencias fueron profundas: más de 50 millones de muertos, la división de Europa, el surgimiento de Estados Unidos y la URSS como superpotencias, el inicio de la Guerra Fría, la creación de las Naciones Unidas, y procesos de descolonización en Asia y África.""",
                    
                    """El Holocausto representa uno de los genocidios más sistemáticos y documentados de la historia. Entre 1933 y 1945, el régimen nazi alemán y sus colaboradores persiguieron y asesinaron sistemáticamente a aproximadamente seis millones de judíos europeos, junto con millones de otras víctimas incluyendo romaníes, personas con discapacidades, prisioneros políticos, y otros grupos considerados "indeseables" por la ideología nazi.

La persecución comenzó gradualmente con las Leyes de Núremberg (1935) que privaron a los judíos de la ciudadanía alemana y prohibieron matrimonios entre judíos y no judíos. La violencia escaló con la Kristallnacht (Noche de los Cristales Rotos) en 1938, cuando sinagogas y negocios judíos fueron atacados en toda Alemania.

Con el inicio de la Segunda Guerra Mundial, los nazis implementaron la "Solución Final", un plan sistemático para el exterminio de todos los judíos europeos. Se construyeron campos de concentración y exterminio en Polonia ocupada, incluyendo Auschwitz-Birkenau, Treblinka, Sobibor, y otros, equipados con cámaras de gas para asesinatos masivos.

El Holocausto no fue solo responsabilidad de líderes nazis, sino que involucró la participación o complicidad de miles de individuos: funcionarios gubernamentales, policías, militares, trabajadores del ferrocarril, y ciudadanos comunes. También hubo resistencia heroica por parte de víctimas en guetos como Varsovia, y personas valientes que arriesgaron sus vidas para salvar a los perseguidos.

La liberación de los campos por las fuerzas aliadas en 1944-1945 reveló al mundo la magnitud de los crímenes. Los Juicios de Núremberg (1945-1946) establecieron precedentes legales importantes para juzgar crímenes contra la humanidad y genocidio.

El legado del Holocausto continúa influyendo en el derecho internacional, la educación sobre derechos humanos, y las políticas para prevenir genocidios. Instituciones como el Museo Memorial del Holocausto de Estados Unidos y Yad Vashem en Israel preservan la memoria y educan a futuras generaciones."""
                ],
                "summary_focus": ["cronología", "causas", "eventos_clave", "consecuencias", "personajes_principales"],
                "key_concepts": ["Segunda Guerra Mundial", "Hitler", "Holocausto", "Nazis", "Aliados", "Eje", "Pearl Harbor", "Stalingrado", "Normandía", "bomba atómica"]
            },
            
            "ciencias_fotosintesis": {
                "texts": [
                    """La fotosíntesis es el proceso biológico fundamental mediante el cual las plantas, algas y ciertas bacterias convierten la energía lumínica del sol en energía química almacenada en moléculas orgánicas. Este proceso es esencial para la vida en la Tierra, ya que produce el oxígeno que respiramos y forma la base de prácticamente todas las cadenas alimenticias.

El proceso ocurre principalmente en las hojas de las plantas, específicamente en orgánulos celulares llamados cloroplastos. Estos contienen un pigmento verde llamado clorofila, que es capaz de absorber la luz solar, particularmente en las longitudes de onda roja y azul del espectro electromagnético.

La fotosíntesis se puede resumir en la ecuación química: 6CO₂ + 6H₂O + energía lumínica → C₆H₁₂O₆ + 6O₂. Esto significa que seis moléculas de dióxido de carbono y seis moléculas de agua, en presencia de luz solar, se convierten en una molécula de glucosa y seis moléculas de oxígeno.

El proceso se divide en dos fases principales: las reacciones dependientes de luz (fotofase) y las reacciones independientes de luz (fase oscura o ciclo de Calvin). Durante la fotofase, que ocurre en los tilacoides de los cloroplastos, la luz solar excita los electrones de la clorofila, iniciando una cadena de reacciones que produce ATP (adenosín trifosfato) y NADPH (nicotinamida adenina dinucleótido fosfato), moléculas que almacenan energía.

En el ciclo de Calvin, que tiene lugar en el estroma de los cloroplastos, el CO₂ atmosférico se "fija" usando la energía del ATP y NADPH producidos en la fotofase. Este proceso regenerativo produce glucosa, que puede ser utilizada inmediatamente para obtener energía o almacenada como almidón.

La fotosíntesis es crucial para el equilibrio de gases en la atmósfera terrestre. Las plantas absorben CO₂ (un gas de efecto invernadero) y liberan O₂, ayudando a regular el clima global. Sin este proceso, la vida aeróbica no podría existir en nuestro planeta.""",
                    
                    """La respiración celular es el proceso complementario a la fotosíntesis mediante el cual las células de todos los organismos vivos descomponen moléculas orgánicas para liberar energía utilizable. Mientras que la fotosíntesis almacena energía en moléculas orgánicas, la respiración celular libera esa energía para actividades vitales.

Este proceso ocurre en las mitocondrias de las células eucariotas y se puede resumir en la ecuación: C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + ATP. La glucosa y el oxígeno se combinan para producir dióxido de carbono, agua y energía en forma de ATP.

La respiración celular consta de tres etapas principales: glucólisis, ciclo de Krebs (o ciclo del ácido cítrico), y cadena de transporte de electrones. La glucólisis ocurre en el citoplasma y descompone la glucosa en piruvato, produciendo una pequeña cantidad de ATP. El ciclo de Krebs tiene lugar en la matriz mitocondrial y completa la oxidación de las moléculas orgánicas. La cadena de transporte de electrones, ubicada en la membrana interna mitocondrial, produce la mayor parte del ATP.

La eficiencia de la respiración celular es notable: puede extraer aproximadamente 32 moléculas de ATP de una sola molécula de glucosa. Esta energía se utiliza para todos los procesos vitales: crecimiento, movimiento, síntesis de proteínas, mantenimiento de la temperatura corporal, y funcionamiento del sistema nervioso.

La fotosíntesis y respiración celular forman un ciclo complementario en la biosfera. Las plantas realizan ambos procesos, pero durante el día, la fotosíntesis predomina, mientras que durante la noche, solo ocurre la respiración. Los animales dependen completamente de la respiración celular y del oxígeno producido por las plantas fotosintéticas."""
                ],
                "summary_focus": ["proceso", "ecuaciones_químicas", "fases", "importancia_ecológica", "orgánulos_celulares"],
                "key_concepts": ["fotosíntesis", "clorofila", "cloroplastos", "ATP", "glucosa", "oxígeno", "dióxido de carbono", "ciclo de Calvin", "respiración celular", "mitocondrias"]
            },
            
            "tecnologia_inteligencia_artificial": {
                "texts": [
                    """La Inteligencia Artificial (IA) es una rama de la ciencia de la computación que se enfoca en crear sistemas capaces de realizar tareas que típicamente requieren inteligencia humana. Estos sistemas pueden aprender, razonar, percibir, procesar lenguaje natural, y tomar decisiones de manera autónoma o semi-autónoma.

La IA moderna se basa principalmente en el aprendizaje automático (machine learning), donde los algoritmos aprenden patrones de grandes conjuntos de datos sin ser programados explícitamente para cada tarea específica. Dentro del machine learning, el aprendizaje profundo (deep learning) utiliza redes neuronales artificiales con múltiples capas para modelar y entender datos complejos.

Las redes neuronales artificiales se inspiran en el funcionamiento del cerebro humano, con nodos interconectados (neuronas artificiales) que procesan y transmiten información. Estas redes pueden reconocer patrones en imágenes, comprender y generar texto, traducir idiomas, y incluso crear contenido original como arte y música.

Las aplicaciones de IA son vastas y crecientes: vehículos autónomos que pueden navegar sin conductor humano, sistemas de recomendación que personalizan contenido en plataformas como Netflix y Spotify, asistentes virtuales como Siri y Alexa, diagnóstico médico automatizado que puede detectar enfermedades en imágenes médicas, traducción automática en tiempo real, y sistemas de reconocimiento facial para seguridad.

En el ámbito empresarial, la IA está transformando industrias enteras. En finanzas, algoritmos de IA detectan fraudes y realizan trading automatizado. En manufactura, optimizan cadenas de suministro y predicen mantenimiento de maquinaria. En marketing, personalizan experiencias de cliente y optimizan campañas publicitarias.

Sin embargo, la IA también presenta desafíos éticos y sociales importantes: preocupaciones sobre privacidad de datos, potencial sesgo en algoritmos, desplazamiento laboral, y la necesidad de regulación apropiada. El desarrollo responsable de IA requiere consideración cuidadosa de estos factores para maximizar beneficios while minimizando riesgos.""",
                    
                    """El procesamiento de lenguaje natural (PLN) es una subdisciplina de la inteligencia artificial que se enfoca en la interacción entre computadoras y lenguaje humano. El objetivo es permitir que las máquinas comprendan, interpreten y generen lenguaje humano de manera útil y significativa.

Los sistemas de PLN enfrentan desafíos únicos porque el lenguaje humano es inherentemente ambiguo, contextual y evolutivo. Una misma palabra puede tener múltiples significados dependiendo del contexto, y el significado puede cambiar con el tiempo. Además, el lenguaje incluye elementos como sarcasmo, metáforas y referencias culturales que son difíciles de codificar algorítmicamente.

Las técnicas modernas de PLN utilizan modelos de transformer y atención, como BERT, GPT, y T5, que han revolucionado el campo. Estos modelos son entrenados en enormes corpus de texto y pueden capturar relaciones complejas entre palabras y conceptos. La arquitectura transformer permite que el modelo "atienda" a diferentes partes del texto simultáneamente, mejorando la comprensión contextual.

Las aplicaciones de PLN incluyen traducción automática que puede manejar múltiples idiomas con precisión creciente, análisis de sentimientos para entender opiniones en redes sociales y reseñas, sistemas de pregunta-respuesta que pueden extraer información específica de documentos largos, generación automática de resúmenes, chatbots inteligentes para atención al cliente, y asistentes virtuales que pueden mantener conversaciones naturales.

En educación, PLN está siendo utilizado para crear sistemas de tutoría adaptativa que pueden proporcionar retroalimentación personalizada, generar preguntas de práctica automáticamente, y evaluar respuestas de estudiantes en formato abierto. También se usa para análisis de textos académicos y detección de plagio.

Los desafíos actuales incluyen mejorar la comprensión de contexto a largo plazo, manejar lenguaje específico de dominio, mantener coherencia en generación de texto largo, y desarrollar sistemas que puedan explicar su razonamiento de manera comprensible para humanos."""
                ],
                "summary_focus": ["definición", "aplicaciones", "tecnologías_clave", "desafíos", "impacto_social"],
                "key_concepts": ["inteligencia artificial", "machine learning", "deep learning", "redes neuronales", "algoritmos", "PLN", "transformer", "aplicaciones", "ética", "automatización"]
            }
        }
    
    def _setup_question_generation_patterns(self):
        """Configura patrones avanzados para generar preguntas contextuales"""
        self.question_patterns = {
            "comprension_basica": [
                "¿Cuál es la definición de {concepto} según el texto?",
                "¿Qué características principales tiene {concepto}?",
                "¿Dónde ocurre el proceso de {proceso}?",
                "¿Cuándo tuvo lugar {evento}?"
            ],
            "analisis_relaciones": [
                "¿Cómo se relaciona {concepto1} con {concepto2}?",
                "¿Qué diferencias existen entre {concepto1} y {concepto2}?",
                "¿Por qué {causa} llevó a {efecto}?",
                "¿Cuál es la importancia de {concepto} en el contexto de {tema}?"
            ],
            "aplicacion_conocimiento": [
                "¿Qué consecuencias tendría si {condicion}?",
                "¿Cómo podrías aplicar {concepto} para resolver {problema}?",
                "¿Qué factores influyen en {proceso}?",
                "¿Por qué es importante {concepto} para {objetivo}?"
            ],
            "evaluacion_critica": [
                "¿Cuáles son las ventajas y desventajas de {concepto}?",
                "¿Qué evidencia apoya la afirmación de que {afirmacion}?",
                "¿Cómo evaluarías la efectividad de {solucion}?",
                "¿Qué alternativas existen a {propuesta}?"
            ]
        }
    
    def _setup_feedback_patterns(self):
        """Configura patrones avanzados para retroalimentación personalizada"""
        self.feedback_patterns = {
            "excelente": {
                "opening": [
                    "¡Extraordinario trabajo! Has demostrado un dominio excepcional de {tema}.",
                    "¡Felicitaciones! Tu comprensión de {tema} es realmente impresionante.",
                    "¡Excelente rendimiento! Has logrado conectar los conceptos de {tema} de manera sobresaliente."
                ],
                "strengths": [
                    "Tu manejo de {concepto1} y {concepto2} demuestra una comprensión profunda del tema",
                    "Has mostrado excelente capacidad para analizar relaciones entre {concepto1} y {concepto2}",
                    "Tu comprensión de las implicaciones de {concepto} es particularmente sólida"
                ],
                "recommendations": [
                    "Para seguir creciendo, te sugiero explorar aspectos más avanzados de {tema}",
                    "Podrías investigar las aplicaciones más recientes de {concepto} en contextos profesionales",
                    "Te recomiendo compartir tu conocimiento tutoreando a otros estudiantes"
                ]
            },
            "bueno": {
                "opening": [
                    "¡Buen trabajo! Has logrado una comprensión sólida de {tema}.",
                    "Tu rendimiento demuestra una base bien establecida en {tema}.",
                    "Has captado correctamente los aspectos fundamentales de {tema}."
                ],
                "areas_to_improve": [
                    "Para mejorar, enfócate en profundizar tu comprensión de {concepto_debil}",
                    "Te beneficiarías de revisar más a fondo {concepto_debil} y su relación con {tema}",
                    "Considera estudiar ejemplos adicionales de {concepto_debil}"
                ],
                "strategies": [
                    "Te sugiero crear mapas conceptuales para visualizar las relaciones entre {concepto1} y {concepto2}",
                    "Practica explicando {concepto_debil} con tus propias palabras",
                    "Busca ejemplos prácticos de {concepto_debil} en situaciones cotidianas"
                ]
            },
            "necesita_mejora": {
                "opening": [
                    "Estás en proceso de construcción de tu comprensión de {tema}. ¡No te desanimes!",
                    "Cada intento es un paso hacia el dominio de {tema}. Tu perseverancia es clave.",
                    "Estás desarrollando tu comprensión de {tema}. El aprendizaje es un proceso gradual."
                ],
                "fundamental_focus": [
                    "Te recomiendo comenzar reforzando los conceptos básicos de {concepto_fundamental}",
                    "Dedica tiempo extra a comprender completamente {concepto_fundamental} antes de avanzar",
                    "Enfócate en dominar {concepto_fundamental}, que es la base para entender {tema}"
                ],
                "study_plan": [
                    "Crea un plan de estudio que incluya revisión diaria de {concepto_fundamental}",
                    "Utiliza recursos visuales como diagramas y videos para reforzar {concepto_fundamental}",
                    "Practica con ejercicios simples de {concepto_fundamental} hasta sentirte cómodo"
                ],
                "motivation": [
                    "Recuerda: cada experto fue una vez principiante. Tu dedicación determinará tu éxito",
                    "El dominio de {tema} requiere tiempo y práctica. Mantén una actitud positiva",
                    "Celebra cada pequeño progreso en tu comprensión de {tema}"
                ]
            }
        }
    
    def generate_comprehensive_summary_data(self, n_samples: int = 600) -> List[Dict[str, Any]]:
        """Genera datos avanzados para fine-tuning de resúmenes"""
        logger.info(f"🔍 Generando {n_samples} ejemplos avanzados de resúmenes")
        
        summary_data = []
        domains = list(self.domain_templates.keys())
        samples_per_domain = n_samples // len(domains)
        
        for domain in domains:
            logger.info(f"📝 Procesando dominio: {domain}")
            domain_data = self.domain_templates[domain]
            
            for i in range(samples_per_domain):
                # Seleccionar texto base
                base_text = random.choice(domain_data["texts"])
                
                # Crear variaciones del texto
                varied_text = self._create_text_variation(base_text, domain)
                
                # Generar resumen educativo de alta calidad
                educational_summary = self._create_advanced_educational_summary(
                    varied_text, domain, domain_data
                )
                
                summary_data.append({
                    "input_text": varied_text,
                    "target_text": educational_summary,
                    "domain": domain,
                    "concepts": domain_data["key_concepts"],
                    "summary_focus": domain_data["summary_focus"],
                    "difficulty": random.choice(["básico", "intermedio", "avanzado"]),
                    "length": "medium"
                })
                
                if (i + 1) % 50 == 0:
                    logger.info(f"   Completados {i + 1}/{samples_per_domain} para {domain}")
        
        # Completar con muestras adicionales si es necesario
        while len(summary_data) < n_samples:
            domain = random.choice(domains)
            domain_data = self.domain_templates[domain]
            base_text = random.choice(domain_data["texts"])
            varied_text = self._create_text_variation(base_text, domain)
            educational_summary = self._create_advanced_educational_summary(
                varied_text, domain, domain_data
            )
            
            summary_data.append({
                "input_text": varied_text,
                "target_text": educational_summary,
                "domain": domain,
                "concepts": domain_data["key_concepts"],
                "summary_focus": domain_data["summary_focus"],
                "difficulty": random.choice(["básico", "intermedio", "avanzado"]),
                "length": "medium"
            })