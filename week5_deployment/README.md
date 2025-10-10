# Week 5: End-to-End NLP Deployment

Deploy production-ready NLP applications with APIs, UIs, and Docker containers.

## 📁 Structure

```
week5_deployment/
├── api/                    # REST APIs
│   ├── sentiment_api.py   # Sentiment analysis API
│   ├── rag_api.py         # RAG system API
│   └── monitoring_api.py  # API with monitoring
├── ui/                     # User Interfaces
│   ├── gradio_app.py      # Gradio interface
│   ├── streamlit_app.py   # Streamlit dashboard
│   └── chatbot_app.py     # RAG chatbot
├── docker/                 # Deployment files
│   ├── Dockerfile
│   └── docker-compose.yml
└── utils/                  # Utilities
    └── logging_config.py
```

## 🚀 Quick Start

### 1. Run Sentiment Analysis API

```bash
cd api
python sentiment_api.py
```

Access at: `http://localhost:8000`
Docs at: `http://localhost:8000/docs`

### 2. Run Gradio UI

```bash
cd ui
python gradio_app.py
```

### 3. Docker Deployment

```bash
cd docker
docker-compose up --build
```

## 📝 API Examples

### Sentiment API

```bash
# Analyze single text
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Batch analysis
curl -X POST "http://localhost:8000/batch-analyze" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible!"]}'
```

### RAG API

```bash
# Add documents
curl -X POST "http://localhost:8001/add-documents" \
  -H "Content-Type: application/json" \
  -d '[{"content": "Python is a programming language"}]'

# Query
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Python?"}'
```

## 🎯 Assignment

Build a complete NLP application:

1. **Backend**: REST API with 3+ endpoints
2. **Frontend**: Gradio or Streamlit UI
3. **Deployment**: Docker configuration
4. **Monitoring**: Logging and metrics
5. **Documentation**: API docs and user guide

**Deliverables:**
- Working API + UI
- Docker files
- README with setup instructions
- Demo video or screenshots
