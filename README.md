# End-to-End NLP Course 2025

A hands-on course covering modern NLP from fundamentals to production deployment using latest APIs and tools.

## 📚 Course Structure

### Week 1: Introduction to NLP (Notebook)
**Location**: `week1_introduction/`

- NLP tasks and preprocessing
- Modern tokenization with transformers
- Prompting and zero-shot inference
- Working with OpenAI, Claude, and Hugging Face

**Run**:
```bash
jupyter notebook week1_introduction/intro_to_nlp.ipynb
```

---

### Week 2: Prompt Engineering & RAG (Notebook)
**Location**: `week2_rag/`

- Advanced prompting techniques
- Building RAG systems with LangChain
- Vector databases (ChromaDB, FAISS)
- Conversational RAG with memory

**Run**:
```bash
jupyter notebook week2_rag/prompt_engineering_and_rag.ipynb
```

**Assignment 1**: Build your own RAG system

---

### Week 3: Fine-tuning (Python Scripts)
**Location**: `week3_finetuning/`

- Traditional fine-tuning
- LoRA (Parameter-Efficient Fine-Tuning)
- Text generation fine-tuning
- Named Entity Recognition

**Scripts**:
```bash
# Traditional fine-tuning
python week3_finetuning/sentiment_finetuning.py --epochs 3

# LoRA fine-tuning (90% fewer parameters!)
python week3_finetuning/lora_finetuning.py --epochs 5

# Text generation
python week3_finetuning/text_generation_lora.py --prompt "Machine learning is"
```

See: [`week3_finetuning/README.md`](week3_finetuning/README.md)

---

### Week 4: Evaluation (Python Scripts)
**Location**: `week4_evaluation/`

- Classification metrics (accuracy, F1, ROC-AUC)
- Sequence metrics (BLEU, ROUGE, METEOR)
- Embedding metrics (BERTScore)
- Model comparison

**Scripts**:
```bash
# Classification metrics
python week4_evaluation/classification_metrics.py --demo

# Sequence metrics (BLEU, ROUGE)
python week4_evaluation/sequence_metrics.py \
  --reference "The cat sat on the mat" \
  --candidate "The cat is on the mat"

# Semantic similarity
python week4_evaluation/embedding_metrics.py \
  --text1 "ML is powerful" \
  --text2 "Machine learning is strong"
```

See: [`week4_evaluation/README.md`](week4_evaluation/README.md)

---

### Week 5: Deployment (Python Scripts)
**Location**: `week5_deployment/`

- REST APIs with FastAPI
- UIs with Gradio/Streamlit
- Docker containerization
- Production best practices

**Run APIs**:
```bash
# Sentiment Analysis API
python week5_deployment/api/sentiment_api.py
# Access: http://localhost:8000/docs

# Gradio UI
python week5_deployment/ui/gradio_app.py
# Access: http://localhost:7860
```

**Docker Deployment**:
```bash
cd week5_deployment/docker
docker-compose up --build
```

See: [`week5_deployment/README.md`](week5_deployment/README.md)

**Assignment 2**: Build end-to-end NLP application

---

## 🚀 Quick Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/e2enlp-2025.git
cd e2enlp-2025
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup API Keys
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 5. Run Setup Script
```bash
bash setup.sh
```

---

## 📖 Repository Structure

```
e2enlp-2025/
├── README.md                          # Main documentation
├── requirements.txt                   # Python dependencies
├── .env.example                       # API key template
├── setup.sh                           # Setup script
│
├── week1_introduction/                # Week 1: Intro (Notebook)
│   └── intro_to_nlp.ipynb
│
├── week2_rag/                         # Week 2: RAG (Notebook)
│   └── prompt_engineering_and_rag.ipynb
│
├── week3_finetuning/                  # Week 3: Fine-tuning (Python)
│   ├── README.md
│   ├── sentiment_finetuning.py
│   ├── lora_finetuning.py
│   └── text_generation_lora.py
│
├── week4_evaluation/                  # Week 4: Evaluation (Python)
│   ├── README.md
│   ├── classification_metrics.py
│   ├── sequence_metrics.py
│   └── embedding_metrics.py
│
└── week5_deployment/                  # Week 5: Deployment (Python)
    ├── README.md
    ├── api/
    │   ├── sentiment_api.py
    │   └── rag_api.py
    ├── ui/
    │   ├── gradio_app.py
    │   └── streamlit_app.py
    └── docker/
        ├── Dockerfile
        └── docker-compose.yml
```

---

## 💡 Learning Path

### Weeks 1-2: Foundations (Notebooks)
Interactive Jupyter notebooks with theory and hands-on examples. Run cell-by-cell to learn concepts.

### Weeks 3-5: Production Skills (Python Scripts)
Production-ready Python scripts you can run, modify, and deploy. Learn by doing.

---

## 🛠️ Technologies

| Technology | Purpose | Used In |
|------------|---------|---------|
| OpenAI GPT-4o-mini | LLM API | Weeks 1-2, 5 |
| Anthropic Claude | LLM API | Weeks 1-2 |
| Hugging Face Transformers | Model library | All weeks |
| LangChain | RAG framework | Week 2, 5 |
| PEFT/LoRA | Fine-tuning | Week 3 |
| FastAPI | REST APIs | Week 5 |
| Gradio/Streamlit | UIs | Week 5 |
| Docker | Deployment | Week 5 |

---

## 📝 Assignments

### Assignment 1: RAG System (Week 2)
Build a RAG system for a specific domain with:
- Document loading and chunking
- Vector store setup
- Question-answering interface
- Evaluation metrics

### Assignment 2: End-to-End App (Week 5)
Create a complete NLP application with:
- Backend REST API
- Frontend UI (Gradio/Streamlit)
- Docker deployment
- Documentation

---

## 🔧 Common Commands

### Run Notebooks
```bash
jupyter notebook
```

### Run Python Scripts
```bash
# With default parameters
python script.py

# With custom parameters
python script.py --epochs 5 --batch-size 16
```

### Run APIs
```bash
# Development
python api_script.py

# Production
uvicorn api_script:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# Stop
docker-compose down
```

---

## 🐛 Troubleshooting

### Installation Issues
```bash
# Upgrade pip
pip install --upgrade pip

# Install specific versions
pip install torch==2.0.0 transformers==4.35.0

# CPU-only PyTorch (if no GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Memory Issues
```bash
# Use smaller models
python script.py --model-name distilbert-base-uncased

# Reduce batch size
python script.py --batch-size 4

# Use LoRA instead of full fine-tuning
python week3_finetuning/lora_finetuning.py
```

### API Key Issues
1. Check `.env` file exists and has correct keys
2. Verify no leading/trailing spaces in keys
3. Ensure environment variables are loaded:
   ```bash
   source .env  # or restart terminal
   ```

---

## 📖 Documentation

- **Course Docs**: See individual week README files
- **OpenAI API**: https://platform.openai.com/docs
- **Anthropic Claude**: https://docs.anthropic.com/
- **LangChain**: https://python.langchain.com/
- **Hugging Face**: https://huggingface.co/docs

---

## 🎯 Learning Outcomes

After completing this course, you will:

✅ Build NLP apps using modern LLMs (OpenAI, Claude)
✅ Implement RAG systems for knowledge retrieval
✅ Fine-tune models with parameter-efficient methods (LoRA)
✅ Evaluate NLP models comprehensively
✅ Deploy production-ready APIs and UIs
✅ Containerize applications with Docker
✅ Follow ML engineering best practices

---

## 🤝 Contributing

Issues and pull requests welcome! Please check existing issues first.

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- OpenAI and Anthropic for LLM APIs
- Hugging Face for transformers
- LangChain community
- All open-source contributors

---

**Happy Learning! 🚀**

For questions, open an issue or refer to individual week README files.
