# README.md para el backend
"""
# IA Educativa Backend

Backend FastAPI para aplicación de contenido educativo con IA generativa.

## Características

- 📄 Procesamiento de archivos PDF y texto
- 🤖 Generación de resúmenes con IA
- 📊 Análisis de conceptos clave y visualizaciones
- ❓ Generación automática de quizzes
- 📈 Retroalimentación pedagógica personalizada
- 🔒 API segura con validación de datos

## Instalación

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
OPENAI_API_KEY=tu_api_key_aqui
SECRET_KEY=tu_secret_key_seguro
ENVIRONMENT=development
```

4. **Ejecutar servidor**:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

### Upload
- `POST /api/v1/upload/upload-file` - Subir archivo PDF/texto
- `POST /api/v1/upload/upload-text` - Enviar texto directo

### Summary
- `POST /api/v1/summary/generate-summary` - Generar resumen
- `POST /api/v1/summary/generate-visualization` - Crear visualizaciones

### Quiz
- `POST /api/v1/quiz/generate-quiz` - Generar quiz
- `POST /api/v1/quiz/submit-quiz` - Enviar respuestas
- `GET /api/v1/quiz/quiz-session/{session_id}` - Info de sesión

## Tecnologías

- **FastAPI** - Framework web
- **OpenAI GPT** - Generación de contenido
- **spaCy** - Procesamiento de lenguaje natural
- **PyPDF2** - Procesamiento de PDFs
- **Pydantic** - Validación de datos

## Estructura del Proyecto

```
backend/
├── app/
│   ├── main.py              # Aplicación principal
│   ├── core/                # Configuración
│   ├── models/              # Modelos de datos
│   ├── services/            # Lógica de negocio
│   ├── api/routes/          # Endpoints
│   └── utils/               # Utilidades
├── uploads/                 # Archivos subidos
└── requirements.txt         # Dependencias
```

## Variables de Entorno

| Variable | Descripción | Requerido |
|----------|-------------|-----------|
| `OPENAI_API_KEY` | API key de OpenAI | Sí |
| `SECRET_KEY` | Clave secreta para JWT | Sí |
| `MAX_FILE_SIZE` | Tamaño máximo de archivo | No |
| `ENVIRONMENT` | Entorno (dev/prod) | No |

## Desarrollo

Para desarrollo local:

1. Clonar repositorio
2. Instalar dependencias
3. Configurar .env
4. Ejecutar `uvicorn app.main:app --reload`
5. Documentación en http://localhost:8000/docs

## Deployment

Para producción:
- Configurar variables de entorno
- Usar servidor ASGI (Gunicorn + Uvicorn)
- Configurar proxy reverso (Nginx)
- Asegurar almacenamiento persistente para uploads

## Licencia

MIT License
"""