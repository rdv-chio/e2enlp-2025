"""
Sentiment Analysis API
Production-ready FastAPI application for sentiment analysis.

Usage:
    python sentiment_api.py
    # or
    uvicorn sentiment_api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List
import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyze sentiment in text using transformer models",
    version="1.0.0"
)

# Global model variable
classifier = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global classifier
    logger.info("Loading sentiment analysis model...")
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    logger.info("Model loaded successfully!")


# Request/Response models
class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

class SentimentOutput(BaseModel):
    text: str
    sentiment: str
    confidence: float


# Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Sentiment Analysis API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/analyze", "/batch-analyze", "/health"]
    }


@app.post("/analyze", response_model=SentimentOutput)
async def analyze_sentiment(input_data: TextInput):
    """Analyze sentiment of a single text."""
    try:
        result = classifier(input_data.text)[0]
        return SentimentOutput(
            text=input_data.text,
            sentiment=result["label"],
            confidence=result["score"]
        )
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-analyze")
async def batch_analyze(input_data: BatchTextInput):
    """Analyze sentiment of multiple texts."""
    try:
        results = classifier(input_data.texts)
        return [
            SentimentOutput(
                text=text,
                sentiment=result["label"],
                confidence=result["score"]
            )
            for text, result in zip(input_data.texts, results)
        ]
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
