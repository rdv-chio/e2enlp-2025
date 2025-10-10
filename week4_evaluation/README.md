# Week 4: Evaluating NLP Models

Comprehensive evaluation techniques for NLP models including classification, generation, and retrieval tasks.

## 📚 Topics Covered

1. **Classification Metrics** - Accuracy, precision, recall, F1, ROC-AUC
2. **Sequence Metrics** - BLEU, ROUGE, METEOR for generation tasks
3. **Embedding Metrics** - BERTScore, semantic similarity
4. **RAG Evaluation** - Context relevance, answer quality

## 📁 Files

- `classification_metrics.py` - Complete classification evaluation
- `sequence_metrics.py` - BLEU, ROUGE, METEOR for text generation
- `embedding_metrics.py` - BERTScore and semantic similarity
- `rag_evaluation.py` - RAG system evaluation framework

## 🚀 Quick Start

### 1. Classification Metrics

```bash
python classification_metrics.py \
    --predictions-file predictions.txt \
    --labels-file labels.txt
```

Or use built-in examples:
```bash
python classification_metrics.py --demo
```

### 2. Sequence Metrics (BLEU, ROUGE)

```bash
python sequence_metrics.py \
    --reference "The cat sat on the mat" \
    --candidate "The cat is on the mat"
```

### 3. Embedding-based Metrics

```bash
python embedding_metrics.py \
    --text1 "The cat sat on the mat" \
    --text2 "A feline rested on the rug"
```

### 4. RAG System Evaluation

```bash
python rag_evaluation.py \
    --questions questions.json \
    --evaluate-all
```

## 📊 Metrics Guide

### Classification Metrics

| Metric | Use Case | Range | Best Value |
|--------|----------|-------|------------|
| Accuracy | Balanced classes | 0-1 | 1.0 |
| Precision | Minimize false positives | 0-1 | 1.0 |
| Recall | Minimize false negatives | 0-1 | 1.0 |
| F1 Score | Balance precision & recall | 0-1 | 1.0 |
| ROC-AUC | Binary classification | 0-1 | 1.0 |

### Sequence Metrics

| Metric | Task | Range | Notes |
|--------|------|-------|-------|
| BLEU | Machine Translation | 0-1 | N-gram overlap |
| ROUGE | Summarization | 0-1 | Recall-oriented |
| METEOR | Translation | 0-1 | Considers synonyms |
| BERTScore | General | 0-1 | Semantic similarity |

### When to Use Which Metric

**Classification:**
- Use **accuracy** for balanced datasets
- Use **F1** for imbalanced datasets
- Use **precision** when false positives are costly
- Use **recall** when false negatives are costly

**Generation:**
- Use **BLEU** for translation
- Use **ROUGE** for summarization
- Use **BERTScore** for semantic quality
- Use **perplexity** for language models

**RAG Systems:**
- **Retrieval quality**: Precision@K, Recall@K
- **Answer quality**: BERTScore, semantic similarity
- **Groundedness**: Check if answer is based on sources

## 💡 Examples

### Evaluate a Sentiment Classifier

```python
from classification_metrics import evaluate_classifier

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 0, 0]

results = evaluate_classifier(y_true, y_pred)
print(results)
```

### Evaluate Translation Quality

```python
from sequence_metrics import evaluate_translation

reference = "The cat is on the mat"
candidate = "The cat sits on the mat"

scores = evaluate_translation(reference, candidate)
print(f"BLEU: {scores['bleu']}")
print(f"ROUGE-L: {scores['rouge_l']}")
```

### Compare Model Outputs

```python
from embedding_metrics import compare_texts

text1 = "Machine learning is powerful"
text2 = "ML is a powerful tool"

similarity = compare_texts(text1, text2)
print(f"Semantic similarity: {similarity}")
```

## 🎯 Assignment

Evaluate your Week 3 fine-tuned model:

1. **Prepare test data** (at least 100 examples)
2. **Generate predictions** from your model
3. **Calculate metrics** using these scripts
4. **Compare** with baseline model
5. **Visualize results** (confusion matrix, ROC curve)
6. **Write report** with insights

**Deliverables:**
- Evaluation scripts
- Results (JSON/CSV format)
- Visualizations
- Comparison report
- Recommendations for improvement

## 📖 Additional Resources

- [Evaluate Library](https://huggingface.co/docs/evaluate/)
- [SciKit-Learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [BLEU Score Explained](https://cloud.google.com/translate/automl/docs/evaluate#bleu)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)
