# Quick Start Guide

Get up and running in 5 minutes!

## 1. Setup (One-time)

```bash
# Clone repo
git clone https://github.com/yourusername/e2enlp-2025.git
cd e2enlp-2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Run setup
bash setup.sh

# Add API keys to .env
cp .env.example .env
# Edit .env with your OpenAI and Anthropic keys
```

## 2. Week-by-Week Guide

### Week 1: Introduction (30 min)
```bash
jupyter notebook week1_introduction/intro_to_nlp.ipynb
```
Learn: NLP basics, tokenization, prompting

### Week 2: RAG (45 min)
```bash
jupyter notebook week2_rag/prompt_engineering_and_rag.ipynb
```
Learn: Advanced prompting, RAG systems, vector stores

**📝 Assignment 1**: Build a RAG system

### Week 3: Fine-tuning (30 min)
```bash
# Run demo
python week3_finetuning/lora_finetuning.py --epochs 3

# See all options
python week3_finetuning/lora_finetuning.py --help
```
Learn: Fine-tuning, LoRA, parameter efficiency

### Week 4: Evaluation (20 min)
```bash
# Run demo
python week4_evaluation/classification_metrics.py --demo

# Try sequence metrics
python week4_evaluation/sequence_metrics.py
```
Learn: Metrics, model comparison, evaluation

### Week 5: Deployment (30 min)
```bash
# Run API
python week5_deployment/api/sentiment_api.py
# Open http://localhost:8000/docs

# Run UI (in new terminal)
python week5_deployment/ui/gradio_app.py
# Open http://localhost:7860
```
Learn: APIs, UIs, Docker, production deployment

**📝 Assignment 2**: Build end-to-end app

## 3. Quick Commands

```bash
# Test everything works
python week4_evaluation/classification_metrics.py --demo

# Run a fine-tuning example
python week3_finetuning/sentiment_finetuning.py --epochs 2

# Start API server
python week5_deployment/api/sentiment_api.py

# Launch Gradio UI
python week5_deployment/ui/gradio_app.py
```

## 4. Common Issues

**Import errors**: `pip install -r requirements.txt`

**API key errors**: Edit `.env` file with your keys

**Memory errors**: Use smaller models or LoRA

**Need help**: Check individual week README files

## Next Steps

1. Complete Week 1 notebook
2. Work through Week 2 with RAG
3. Try fine-tuning scripts in Week 3
4. Evaluate your models in Week 4
5. Deploy your app in Week 5

**Full documentation**: See [README.md](README.md)
