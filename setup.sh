#!/bin/bash

# E2E NLP Course Setup Script

echo "=================================="
echo "  E2E NLP Course 2025 Setup"
echo "=================================="
echo

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python 3.9+
if ! python -c 'import sys; exit(0 if sys.version_info >= (3,9) else 1)'; then
    echo "❌ Error: Python 3.9 or higher is required"
    exit 1
fi

echo "✅ Python version OK"
echo

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✅ Virtual environment created"
    echo
    echo "Please activate it:"
    echo "  source venv/bin/activate  # Mac/Linux"
    echo "  venv\\Scripts\\activate     # Windows"
    echo
    echo "Then run this script again."
    exit 0
fi

echo "✅ Virtual environment active"
echo

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✅ Pip upgraded"
echo

# Install requirements
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Error installing dependencies"
    exit 1
fi

echo

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab')"
echo "✅ NLTK data downloaded"
echo

# Check .env file
if [ ! -f .env ]; then
    echo "⚠️  .env file not found"
    echo "Creating .env from template..."
    cp .env.example .env
    echo "✅ .env created"
    echo
    echo "⚠️  IMPORTANT: Edit .env and add your API keys:"
    echo "  - OPENAI_API_KEY"
    echo "  - ANTHROPIC_API_KEY"
    echo
else
    echo "✅ .env file exists"
    echo
fi

# Create directories for outputs
echo "Creating output directories..."
mkdir -p results logs data models checkpoints
echo "✅ Directories created"
echo

# Test imports
echo "Testing imports..."
python -c "
import torch
import transformers
import langchain
import openai
import anthropic
import gradio
import fastapi
print('✅ All core packages imported successfully')
"

if [ $? -eq 0 ]; then
    echo
    echo "=================================="
    echo "  Setup Complete! 🎉"
    echo "=================================="
    echo
    echo "Next steps:"
    echo "1. Edit .env with your API keys"
    echo "2. Start with Week 1:"
    echo "   jupyter notebook week1_introduction/intro_to_nlp.ipynb"
    echo
    echo "Quick test:"
    echo "   python week4_evaluation/classification_metrics.py --demo"
    echo
    echo "For help, see README.md or individual week READMEs"
    echo
else
    echo "❌ Error testing imports"
    echo "Please check the error messages above"
    exit 1
fi
