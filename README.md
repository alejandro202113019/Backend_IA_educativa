# IA Educativa Backend - Versión 100% Gratuita

Backend FastAPI para aplicación de contenido educativo con IA generativa **completamente gratuita**.

## 🎉 Modelos Gratuitos Utilizados

- **Resúmenes**: BART (facebook/bart-large-cnn)
- **Quiz Generator**: T5 (google/flan-t5-base) 
- **Análisis**: RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Retroalimentación**: T5 (google/flan-t5-base)

## ✨ Características

- 📄 Procesamiento de archivos PDF y texto
- 🤖 Generación de resúmenes con BART (sin costo)
- 📊 Análisis de conceptos clave con Transformers
- ❓ Generación automática de quizzes con T5
- 📈 Retroalimentación pedagógica personalizada
- 🔒 API segura con validación de datos
- 💰 **100% GRATUITO - Sin APIs de pago**

## 🚀 Instalación Rápida

1. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

2. **Instalar modelo de spaCy en español**:
```bash
python -m spacy download es_core_news_sm
```

3. **Configurar variables de entorno** (.env):
```env
SECRET_KEY=tu_secret_key_seguro
ENVIRONMENT=development
AI_USE_GPU=true
AI_MODEL_SIZE=base
```

4. **Ejecutar servidor**:
```bash
uvicorn app.main:app --reload
```

## 📋 API Endpoints

### Upload
- `POST /api/v1/upload/upload-file` - Subir archivo PDF/texto
- `POST /api/v1/upload/upload-text` - Enviar texto directo

### Summary
- `POST /api/v1/summary/generate-summary` - Generar resumen (BART)
- `POST /api/v1/summary/generate-visualization` - Crear visualizaciones

### Quiz
- `POST /api/v1/quiz/generate-quiz` - Generar quiz (T5)
- `POST /api/v1/quiz/submit-quiz` - Enviar respuestas
- `GET /api/v1/quiz/quiz-session/{session_id}` - Info de sesión

## 🛠 Tecnologías

- **FastAPI** - Framework web
- **BART** - Generación de resúmenes (Facebook AI)
- **T5** - Generación de preguntas (Google)
- **RoBERTa** - Análisis de sentimientos (Cardiff NLP)
- **Transformers** - Biblioteca de Hugging Face
- **PyTorch** - Framework de deep learning
- **spaCy** - Procesamiento de lenguaje natural
- **PyPDF2** - Procesamiento de PDFs

## 🎯 Ventajas de esta Versión

| Característica | Antes (OpenAI) | Ahora (Gratuito) |
|---|---|---|
| **Costo** | $0.002/1K tokens | ✅ $0 - Totalmente gratis |
| **Límites** | Rate limits | ✅ Sin límites |
| **Privacidad** | Datos enviados a OpenAI | ✅ Todo local |
| **Latencia** | Depende de API | ✅ Procesamiento local |
| **Dependencias** | Internet requerido | ✅ Funciona offline |

## 📊 Rendimiento

- **Resúmenes**: BART genera resúmenes de alta calidad comparables a GPT-3.5
- **Quiz**: T5 crea preguntas coherentes y educativas
- **Velocidad**: ~2-5 segundos por operación (con GPU)
- **Memoria**: ~2GB RAM mínimo, 4GB recomendado

## 🔧 Configuración Avanzada

### Para usar GPU (recomendado):
```bash
# Instalar PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Para modo CPU únicamente:
```env
AI_USE_GPU=false
AI_MODEL_SIZE=small
```

## 🐳 Docker (Opcional)

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📁 Estructura del Proyecto

```
backend/
├── app/
│   ├── main.py              # Aplicación principal
│   ├── core/
│   │   ├── config.py        # Configuración (sin OpenAI)
│   │   └── security.py      # Seguridad
│   ├── models/              # Modelos de datos
│   ├── services/
│   │   ├── ai_service.py    # BART + T5 + Transformers
│   │   ├── nlp_service.py   # spaCy + análisis
│   │   ├── pdf_processor.py # Procesamiento PDFs
│   │   └── quiz_generator.py # Gestión de quizzes
│   ├── api/routes/          # Endpoints
│   └── utils/               # Utilidades
├── model_cache/             # Caché de modelos descargados
├── uploads/                 # Archivos subidos
└── requirements.txt         # Dependencias actualizadas
```

## 🚀 Migración desde OpenAI

### Pasos para migrar:

1. **Actualizar requirements.txt** con las nuevas dependencias
2. **Reemplazar ai_service.py** con la versión de modelos gratuitos
3. **Actualizar config.py** para remover OPENAI_API_KEY
4. **Ejecutar la primera vez** (descargará modelos automáticamente):

```bash
python -c "from app.services.ai_service import AIService; AIService()"
```

## 💾 Espacio en Disco

Los modelos requieren espacio de descarga inicial:

- **BART (resúmenes)**: ~1.6GB
- **T5-base (quiz)**: ~900MB  
- **RoBERTa (análisis)**: ~500MB
- **Total aproximado**: ~3GB

## ⚡ Optimizaciones

### Para hardware limitado:
```env
AI_MODEL_SIZE=small  # Usa modelos más pequeños
AI_USE_GPU=false     # Modo CPU únicamente
```

### Para mejor rendimiento:
```env
AI_MODEL_SIZE=large  # Modelos más grandes y precisos
AI_USE_GPU=true      # Usa GPU si está disponible
```

## 🔍 Comparación de Modelos

| Modelo | Tamaño | Calidad | Velocidad | Memoria |
|--------|--------|---------|-----------|---------|
| **T5-small** | 242MB | ⭐⭐⭐ | ⚡⚡⚡ | 1GB |
| **T5-base** | 892MB | ⭐⭐⭐⭐ | ⚡⚡ | 2GB |
| **BART-large** | 1.6GB | ⭐⭐⭐⭐⭐ | ⚡ | 4GB |

## 🧪 Testing

```bash
# Probar el servicio de IA
python -m pytest tests/test_ai_service.py

# Probar endpoints
python -m pytest tests/test_endpoints.py

# Benchmark de rendimiento
python scripts/benchmark_models.py
```

## 🔧 Solución de Problemas

### Error: "CUDA out of memory"
```env
AI_USE_GPU=false
AI_MODEL_SIZE=small
```

### Error: "Model not found"
```bash
# Limpiar caché y volver a descargar
rm -rf model_cache/
python -c "from app.services.ai_service import AIService; AIService()"
```

### Lentitud en CPU
```bash
# Instalar optimizaciones para CPU
pip install intel-extension-for-pytorch  # Para CPUs Intel
# o
pip install torch-audio --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📈 Monitoreo

El sistema incluye logs detallados:

```bash
# Ver logs en tiempo real
tail -f app.log

# Buscar errores específicos
grep "ERROR" app.log
```

## 🆙 Actualizaciones Futuras

- [ ] Integración con Stable Diffusion para generar imágenes
- [ ] Soporte para modelos Mistral y LLaMA 2
- [ ] Optimizaciones adicionales para ARM (Apple Silicon)
- [ ] Cache inteligente de resultados
- [ ] API de batch processing

## 📞 Soporte

- **Documentación**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health  
- **Logs**: Revisar logs de la aplicación
- **Issues**: Crear issue en el repositorio

## 📄 Licencia

MIT License - Uso libre y gratuito