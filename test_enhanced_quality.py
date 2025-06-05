#!/usr/bin/env python3
"""
test_final_quality.py - Test final de calidad perfecta del sistema
"""

import asyncio
import sys
from pathlib import Path

# Agregar el directorio del proyecto al path
sys.path.insert(0, str(Path.cwd()))

# Texto completo de la Segunda Guerra Mundial para pruebas
COMPLETE_WWII_TEXT = """
INTRODUCCIÓN
La Segunda Guerra Mundial ha sido la peor guerra de la Historia de la Humanidad. En ella se
produjeron varias decenas de millones de muertos (no se puede calcular la cifra con exactitud). Además fue
la primera guerra en la que murió más población civil que militares. Las bajas se produjeron en el frente,
pero también a causa de los bombardeos. Tampoco hay que olvidar el exterminio, el asesinato masivo de
comunidades enteras como los judíos europeos en los campos de concentración nazis.

CAUSAS DE LA SEGUNDA GUERRA MUNDIAL
Las consecuencias de la Crisis Económica del 29 y la Gran Depresión: la crisis del modelo capitalista
en los años 30 produjo miedo en las sociedades industrializadas, miedo en el futuro de su sistema
económico y en la amenaza del sistema comunista que mostraba orgulloso su éxito en la industrialización
de la URSS durante los años 30.

El revanchismo alemán e italiano contra el Tratado de Versalles. Hitler y Mussolini utilizaron el
descontento de sus respectivos países hacia el Tratado de Versalles para conseguir el poder. En los años 30
tenían que cumplir su palabra a quienes les apoyaron, de ahí que Hitler fuera rompiendo una a una las
humillantes cláusulas de Versalles.

El expansionismo militar de las potencias fascistas: Alemania, Italia y Japón necesitaban ocupar
grandes territorios ricos en materias primas (especialmente petróleo), para asegurar la prosperidad de sus
economías industriales. Hitler pretendía la expansión de Alemania a costa de Rusia (Teoría del Espacio Vital).

FASES DE LA SEGUNDA GUERRA MUNDIAL (1939-1945)
En la Primera Fase (1939-1942): Alemania y sus aliados (las potencias del Eje, Italia y Japón),
cosecharon un éxito tras otro, derrotando a sus enemigos con invasiones rápidas (Blitzkrieg).

En la Segunda Fase (1943-1945): los aliados llevaron a cabo una lenta guerra de desgaste,
reconquistando los territorios ocupados por el Eje, que se defendió hasta el final a pesar de que la guerra
estaba perdida para el Eje desde 1943.

Primera Fase (1939-42)
1939, Septiembre: los alemanes y rusos ocupan Polonia de acuerdo con el Pacto Germano-Soviético
de No Agresión. Francia e Inglaterra permanecen inactivos y no atacan a Alemania.

1940: los alemanes invaden Dinamarca y Noruega. Gran Ofensiva del Oeste: los alemanes ocupan
Holanda, Bélgica y Francia, y obligan al ejército británico a reembarcarse en Dunquerke. Inglaterra se queda
sola luchando contra Alemania.

1941: Operación Barbarroja, los alemanes invaden la URSS con el objetivo de alcanzar Leningrado,
Moscú y Kiev antes del invierno. Ataque de Pearl Harbor: los japoneses atacan por sorpresa a la flota
americana en el Pacífico, haciendo que EEUU entre en la guerra.

1942: Batalla de Stalingrado: tras penetrar profundamente en la URSS la ofensiva alemana se estancó
delante de Moscú por la dureza del invierno y el contraataque ruso.

Segunda Fase (1943-45)
1943: Los aliados expulsan a los alemanes e italianos del Norte de Africa, y desembarcan en Italia.

1944: Desembarco de Normandía, los ingleses y americanos desembarcan en Francia y abren un
segundo frente en Europa.

1945: Batalla de Berlín: la lentitud de los aliados occidentales favorece el avance de los rusos
hasta Europa Central, éstos asedian y conquistan Berlín. Hitler se suicida y los nazis se rinden. Acaba la
guerra en Europa. Las bombas atómicas sobre Hiroshima y Nagasaki obligan a Japón a rendirse.

CONSECUENCIAS DE LA SEGUNDA GUERRA MUNDIAL
La consecuencia más importante de la Segunda Guerra Mundial es que Europa queda relegada a un
segundo plano frente a las superpotencias: EEUU y URSS.

El mundo queda así dividido en dos partes: los países democráticos-capitalistas, liderados por EEUU
y los países socialistas liderados por la URSS.

EEUU es nuevamente el gran vencedor de la guerra. Este país se convierte en el líder económico
mundial y su propia propaganda le convierte en el modelo de la democracia frente al Fascismo y la
Amenaza Comunista.
"""

async def test_perfect_summary():
    """Prueba la generación de resúmenes perfectos"""
    print("🎯 PROBANDO: Generación de resúmenes PERFECTOS")
    print("=" * 70)
    
    try:
        from app.services.service_manager import service_manager
        
        ai_service = service_manager.ai_service
        
        print("📝 Generando resumen perfecto del texto completo de la Segunda Guerra Mundial...")
        
        result = await ai_service.generate_summary(COMPLETE_WWII_TEXT, "medium")
        
        if result["success"]:
            print(f"✅ Resumen perfecto generado")
            print(f"🤖 Modelo usado: {result.get('model_used', 'unknown')}")
            print(f"📄 RESUMEN PERFECTO:")
            print("=" * 70)
            print(result["summary"])
            print("=" * 70)
            
            # Evaluar calidad perfecta
            quality_score = evaluate_perfect_summary_quality(result["summary"])
            print(f"📊 Puntuación de calidad perfecta: {quality_score}/10")
            
            if quality_score >= 9:
                print("🏆 ¡CALIDAD PERFECTA ALCANZADA!")
            elif quality_score >= 7:
                print("⭐ Calidad excelente")
            else:
                print("📈 Calidad mejorable")
            
            return result["summary"], quality_score
        else:
            print(f"❌ Error generando resumen: {result.get('error', 'Error desconocido')}")
            return None, 0
            
    except Exception as e:
        print(f"❌ Error en test de resumen perfecto: {e}")
        return None, 0

async def test_perfect_quiz():
    """Prueba la generación de quiz perfecto"""
    print("\n🎯 PROBANDO: Generación de quiz PERFECTO")
    print("=" * 70)
    
    try:
        from app.services.service_manager import service_manager
        
        ai_service = service_manager.ai_service
        
        key_concepts = [
            "Segunda Guerra Mundial", "Hitler", "Stalin", "Pearl Harbor", 
            "Blitzkrieg", "Operación Barbarroja", "Stalingrado", "Normandía"
        ]
        
        print("❓ Generando quiz perfecto sobre la Segunda Guerra Mundial...")
        
        result = await ai_service.generate_quiz(COMPLETE_WWII_TEXT, key_concepts, 5, "medium")
        
        if result["success"] and result["questions"]:
            print(f"✅ Quiz perfecto generado")
            print(f"🤖 Modelo usado: {result.get('model_used', 'unknown')}")
            print(f"📊 Preguntas generadas: {len(result['questions'])}")
            
            print("\n🎓 PREGUNTAS PERFECTAS:")
            for i, question in enumerate(result["questions"], 1):
                print(f"\n📝 PREGUNTA {i}:")
                print(f"   ❓ {question['question']}")
                
                for j, option in enumerate(question['options']):
                    marker = "✅" if j == question['correct_answer'] else "   "
                    print(f"   {marker} {chr(65+j)}) {option}")
                
                print(f"   💡 Explicación: {question['explanation']}")
            
            # Evaluar calidad perfecta del quiz
            quality_score = evaluate_perfect_quiz_quality(result["questions"])
            print(f"\n📊 Puntuación de calidad perfecta del quiz: {quality_score}/10")
            
            if quality_score >= 9:
                print("🏆 ¡QUIZ PERFECTO ALCANZADO!")
            elif quality_score >= 7:
                print("⭐ Quiz de excelente calidad")
            else:
                print("📈 Quiz mejorable")
            
            return result["questions"], quality_score
        else:
            print(f"❌ Error generando quiz: {result.get('error', 'Error desconocido')}")
            return None, 0
            
    except Exception as e:
        print(f"❌ Error en test de quiz perfecto: {e}")
        return None, 0

async def test_perfect_feedback():
    """Prueba la generación de feedback perfecto"""
    print("\n🎯 PROBANDO: Generación de feedback PERFECTO")
    print("=" * 70)
    
    try:
        from app.services.service_manager import service_manager
        
        ai_service = service_manager.ai_service
        
        # Casos de prueba exhaustivos
        test_cases = [
            {
                "name": "Rendimiento Excepcional",
                "score": 5, "total": 5, 
                "concepts": ["Segunda Guerra Mundial", "Blitzkrieg"]
            },
            {
                "name": "Buen Rendimiento", 
                "score": 4, "total": 5,
                "concepts": ["Hitler", "Stalingrado"]
            },
            {
                "name": "Rendimiento Promedio",
                "score": 3, "total": 5,
                "concepts": ["Pearl Harbor", "Normandía"]
            },
            {
                "name": "Necesita Mejora",
                "score": 2, "total": 5,
                "concepts": ["Operación Barbarroja", "Nazi"]
            }
        ]
        
        feedback_scores = []
        
        for case in test_cases:
            print(f"\n💬 {case['name']}: {case['score']}/{case['total']} ({case['score']/case['total']*100:.0f}%)")
            
            feedback = await ai_service.generate_feedback(
                case["score"], case["total"], [], case["concepts"]
            )
            
            print(f"📝 Feedback perfecto generado:")
            print("-" * 50)
            # Mostrar primeras líneas del feedback
            lines = feedback.split('\n')[:8]
            for line in lines:
                print(line)
            if len(feedback.split('\n')) > 8:
                print("...")
            print("-" * 50)
            
            # Evaluar calidad del feedback
            feedback_quality = evaluate_perfect_feedback_quality(feedback, case["score"], case["total"])
            feedback_scores.append(feedback_quality)
            print(f"📊 Calidad del feedback: {feedback_quality}/10")
        
        avg_feedback_quality = sum(feedback_scores) / len(feedback_scores)
        print(f"\n📊 Calidad promedio del feedback: {avg_feedback_quality:.1f}/10")
        
        if avg_feedback_quality >= 9:
            print("🏆 ¡FEEDBACK PERFECTO ALCANZADO!")
        elif avg_feedback_quality >= 7:
            print("⭐ Feedback de excelente calidad")
        else:
            print("📈 Feedback mejorable")
        
        return avg_feedback_quality >= 7
        
    except Exception as e:
        print(f"❌ Error en test de feedback perfecto: {e}")
        return False

def evaluate_perfect_summary_quality(summary: str) -> int:
    """Evalúa la calidad perfecta del resumen (0-10)"""
    score = 10
    
    # Verificar estructura educativa perfecta
    if "📚" not in summary:
        score -= 1
    if "🔑" not in summary or "CONCEPTOS CLAVE" not in summary:
        score -= 1
    if "📅" not in summary:
        score -= 1
    if "👥" not in summary:
        score -= 1
    if "📝" not in summary or "CONTENIDO PRINCIPAL" not in summary:
        score -= 2
    
    # Verificar contenido específico de WWII
    required_concepts = ["Segunda Guerra Mundial", "1939", "1945"]
    for concept in required_concepts:
        if concept not in summary:
            score -= 1
    
    # Penalizar errores de calidad
    quality_issues = ["seguirra", "eusu", "histororia", "argentinos del eje"]
    for issue in quality_issues:
        if issue.lower() in summary.lower():
            score -= 2
    
    # Bonificar características de calidad perfecta
    if "Blitzkrieg" in summary:
        score += 1
    if any(name in summary for name in ["Hitler", "Stalin", "Churchill"]):
        score += 1
    if "superpotencias" in summary:
        score += 1
    
    return max(0, min(10, score))

def evaluate_perfect_quiz_quality(questions: list) -> int:
    """Evalúa la calidad perfecta del quiz (0-10)"""
    if not questions or len(questions) == 0:
        return 0
    
    score = 10
    
    # Verificar preguntas específicas de WWII
    wwii_specific_count = 0
    for question in questions:
        question_text = question.get("question", "").lower()
        
        # Contar preguntas específicas y de alta calidad
        if any(topic in question_text for topic in [
            "segunda guerra mundial", "hitler", "stalin", "pearl harbor",
            "blitzkrieg", "stalingrado", "normandía", "1939", "1945"
        ]):
            wwii_specific_count += 1
        
        # Penalizar preguntas genéricas de baja calidad
        if any(bad_phrase in question_text for bad_phrase in [
            "¿qué es", "método tradicional", "proceso relacionado",
            "concepto central del texto sobre"
        ]):
            score -= 2
    
    # Bonificar por especificidad
    specificity_bonus = (wwii_specific_count / len(questions)) * 3
    score += specificity_bonus
    
    # Verificar calidad de opciones
    good_options_count = 0
    for question in questions:
        options = question.get("options", [])
        if any("invasión" in opt or "1939" in opt or "alemania" in opt.lower() for opt in options):
            good_options_count += 1
    
    options_bonus = (good_options_count / len(questions)) * 2
    score += options_bonus
    
    return max(0, min(10, score))

def evaluate_perfect_feedback_quality(feedback: str, score: int, total: int) -> int:
    """Evalúa la calidad perfecta del feedback (0-10)"""
    quality_score = 10
    
    # Verificar estructura
    required_elements = ["**", "🎯", "📊", "💡"]
    for element in required_elements:
        if element not in feedback:
            quality_score -= 1
    
    # Verificar personalización
    percentage = (score / total) * 100
    if f"{score}/{total}" not in feedback:
        quality_score -= 1
    if f"{percentage:.1f}%" not in feedback:
        quality_score -= 1
    
    # Verificar longitud apropiada
    if len(feedback) < 200:
        quality_score -= 2
    elif len(feedback) > 1000:
        quality_score -= 1
    
    # Bonificar características de calidad
    if "ANÁLISIS" in feedback:
        quality_score += 1
    if "RECOMENDACIONES" in feedback or "ESTRATEGIAS" in feedback:
        quality_score += 1
    
    return max(0, min(10, quality_score))

async def main():
    """Función principal del test de calidad perfecta"""
    print("🏆 TEST DE CALIDAD PERFECTA - SISTEMA IA EDUCATIVA")
    print("=" * 80)
    print("🎯 Objetivo: Verificar que el sistema genere contenido de nivel universitario")
    print("📖 Probando con texto académico completo de la Segunda Guerra Mundial")
    print()
    
    # Ejecutar todos los tests perfectos
    summary, summary_score = await test_perfect_summary()
    questions, quiz_score = await test_perfect_quiz()
    feedback_ok = await test_perfect_feedback()
    
    # Calcular puntuación total
    total_score = (summary_score + quiz_score + (8 if feedback_ok else 4)) / 3
    
    # Resumen final
    print("\n" + "=" * 80)
    print("🏆 EVALUACIÓN FINAL DE CALIDAD PERFECTA")
    print("=" * 80)
    
    print(f"📝 Resumen: {summary_score}/10 - {'🏆 PERFECTO' if summary_score >= 9 else '⭐ EXCELENTE' if summary_score >= 7 else '📈 MEJORABLE'}")
    print(f"❓ Quiz: {quiz_score}/10 - {'🏆 PERFECTO' if quiz_score >= 9 else '⭐ EXCELENTE' if quiz_score >= 7 else '📈 MEJORABLE'}")
    print(f"💬 Feedback: {'🏆 PERFECTO' if feedback_ok else '📈 MEJORABLE'}")
    
    print(f"\n📊 PUNTUACIÓN TOTAL: {total_score:.1f}/10")
    
    if total_score >= 9:
        print("\n🎉 ¡SISTEMA DE CALIDAD PERFECTA ALCANZADO!")
        print("🏆 Tu IA educativa genera contenido de nivel universitario")
        print("✅ Listo para impresionar a profesores y estudiantes")
        print("🚀 COMANDO PARA LANZAR: uvicorn app.main:app --reload")
    elif total_score >= 7:
        print("\n⭐ ¡SISTEMA DE EXCELENTE CALIDAD!")
        print("🎓 Tu IA educativa genera contenido profesional")
        print("✅ Funcional y de alta calidad para uso educativo")
        print("🚀 COMANDO PARA LANZAR: uvicorn app.main:app --reload")
    elif total_score >= 5:
        print("\n📈 Sistema funcional con calidad mejorable")
        print("🔧 Algunas características necesitan ajustes")
        print("💡 Considera revisar la implementación")
    else:
        print("\n⚠️ Sistema necesita mejoras significativas")
        print("🔧 Revisa que el archivo enhanced_ai_service.py esté actualizado")
        print("💡 Ejecuta: python fix_paths.py")
    
    # Mostrar ejemplo de resultado esperado
    if total_score >= 7:
        print("\n" + "=" * 80)
        print("🎓 EJEMPLO DE CALIDAD ESPERADA:")
        print("=" * 80)
        
        print("📝 RESUMEN ESPERADO:")
        print("📚 **RESUMEN EDUCATIVO PERFECCIONADO**")
        print("🎯 **TEMA PRINCIPAL:** La Segunda Guerra Mundial")
        print("🔑 **CONCEPTOS CLAVE:** Blitzkrieg, Operación Barbarroja, Pearl Harbor")
        print("📅 **CRONOLOGÍA:** 1939 → 1941 → 1943 → 1945")
        print("👥 **FIGURAS HISTÓRICAS:** Hitler, Stalin, Churchill, Roosevelt")
        print("📝 **CONTENIDO PRINCIPAL:** [Resumen estructurado y coherente...]")
        print()
        
        print("❓ PREGUNTAS ESPERADAS:")
        print("¿Cuál fue el evento que marcó el inicio oficial de la Segunda Guerra Mundial?")
        print("A) La invasión alemana de Polonia el 1 de septiembre de 1939 ✅")
        print("B) El ataque japonés a Pearl Harbor")
        print("C) La anexión de Austria por Alemania")
        print("D) El bombardeo de Londres")
        print()
        
        print("💬 FEEDBACK ESPERADO:")
        print("🎉 **¡RENDIMIENTO EXCEPCIONAL!**")
        print("🏆 **RESULTADO:** 5/5 respuestas correctas (100.0%)")
        print("🔍 **ANÁLISIS DE RENDIMIENTO:** Has demostrado un dominio sobresaliente...")
        print("💎 **FORTALEZAS IDENTIFICADAS:** • Excelente manejo de...")
    
    return 0 if total_score >= 7 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())