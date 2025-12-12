# Sophia Search — Sistema RAG de Noticias de Chile

Sistema de Retrieval-Augmented Generation (RAG) para consultar noticias chilenas (septiembre 2025) en lenguaje natural.

## Requisitos

- Python 3.12+
- Linux (probado en Ubuntu)

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Railer03/TAL-PROYECTO.git
   cd TAL-PROYECTO
   ```

2. Crea y activa un entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Instala las dependencias:
   ```bash
   pip install pandas spacy sentence-transformers faiss-cpu numpy openai google-generativeai groq python-dotenv tqdm
   python -m spacy download es_core_news_sm
   ```

4. Configura las API keys:
   ```bash
   cp .env.example .env
   ```
   Edita `.env` y reemplaza con tus claves reales.

## Obtener API Keys (gratis)

| Proveedor | Modelo | Enlace |
|-----------|--------|--------|
| Google Gemini | `gemini-2.5-flash` | https://aistudio.google.com/app/apikey |
| Groq | `llama-3.3-70b-versatile` | https://console.groq.com/keys |

## Ejecución

### Modo chat interactivo
```bash
python tal_news_rag/main.py
```

### Modo one-shot (consulta directa)
```bash
python tal_news_rag/main.py "¿Qué pasó en La Serena?"
```

## Estructura del proyecto

| Archivo | Descripción |
|---------|-------------|
| `main.py` | Orquestador principal y CLI |
| `ingestion.py` | Carga CSV, limpieza y NER para extraer regiones |
| `indexing.py` | Embeddings con sentence-transformers + índice FAISS |
| `search.py` | Agente de consulta con filtros (región/fecha) |
| `generation.py` | Generación con Gemini (respaldo Groq) y citas |

## Ejemplo de uso

```
Ingresa tu pregunta: incendio en La Serena

RESPUESTA:
Se produjo un incendio en La Serena en el marco de un "banderazo"...

Fuentes:
[1] emol, Sep 24 2025, "Banderazo" de hinchas de la U termina en desmanes...
```

## Licencia

Proyecto académico — Universidad Austral de Chile, TAL 2025.
