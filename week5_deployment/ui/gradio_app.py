"""
Gradio NLP Multi-Tool
Interactive UI for multiple NLP tasks.

Usage:
    python gradio_app.py
"""

import gradio as gr
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models
logger.info("Loading models...")
sentiment_model = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("question-answering")
logger.info("Models loaded!")


def analyze_sentiment(text):
    """Analyze sentiment."""
    if not text:
        return "Please enter some text."
    result = sentiment_model(text)[0]
    return f"**Sentiment:** {result['label']}\\n**Confidence:** {result['score']:.2%}"


def summarize_text(text, max_length=130):
    """Summarize text."""
    if not text or len(text.split()) < 50:
        return "Text too short for summarization (minimum 50 words)"
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']


def answer_question(context, question):
    """Answer question from context."""
    if not context or not question:
        return "Please provide both context and question."
    result = qa_model(question=question, context=context)
    return f"**Answer:** {result['answer']}\\n**Confidence:** {result['score']:.2%}"


# Create interface
with gr.Blocks(title="NLP Multi-Tool", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 NLP Multi-Tool")
    gr.Markdown("Perform various NLP tasks using state-of-the-art models")

    with gr.Tab("📊 Sentiment Analysis"):
        with gr.Row():
            sentiment_input = gr.Textbox(lines=5, placeholder="Enter text here...", label="Text")
            sentiment_output = gr.Textbox(label="Result")
        sentiment_btn = gr.Button("Analyze Sentiment", variant="primary")
        sentiment_btn.click(fn=analyze_sentiment, inputs=sentiment_input, outputs=sentiment_output)

        gr.Examples(
            examples=[
                ["I love this product! It's amazing!"],
                ["This is terrible. I hate it."],
                ["It's okay, nothing special."]
            ],
            inputs=sentiment_input
        )

    with gr.Tab("📝 Summarization"):
        with gr.Row():
            summary_input = gr.Textbox(lines=10, placeholder="Enter long text to summarize...", label="Text")
            summary_output = gr.Textbox(lines=5, label="Summary")
        max_length = gr.Slider(50, 200, value=130, label="Max Summary Length")
        summary_btn = gr.Button("Summarize", variant="primary")
        summary_btn.click(fn=summarize_text, inputs=[summary_input, max_length], outputs=summary_output)

    with gr.Tab("❓ Question Answering"):
        context_input = gr.Textbox(lines=7, placeholder="Enter context...", label="Context")
        question_input = gr.Textbox(lines=2, placeholder="Ask a question...", label="Question")
        qa_output = gr.Textbox(label="Answer")
        qa_btn = gr.Button("Get Answer", variant="primary")
        qa_btn.click(fn=answer_question, inputs=[context_input, question_input], outputs=qa_output)

        gr.Examples(
            examples=[
                ["The Eiffel Tower is in Paris, France. It was completed in 1889.", "Where is the Eiffel Tower?"],
                ["Python is a high-level programming language created by Guido van Rossum.", "Who created Python?"]
            ],
            inputs=[context_input, question_input]
        )

    gr.Markdown("---")
    gr.Markdown("Built with [Gradio](https://gradio.app) and [Hugging Face Transformers](https://huggingface.co/transformers)")


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
