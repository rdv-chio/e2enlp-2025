# Week 3: Fine-tuning NLP Models

This week covers fine-tuning techniques for NLP models, including traditional fine-tuning and parameter-efficient methods like LoRA.

## 📚 Topics Covered

1. **Traditional Fine-tuning** - Full model fine-tuning for sentiment analysis
2. **LoRA Fine-tuning** - Parameter-efficient fine-tuning with Low-Rank Adaptation
3. **Text Generation** - Fine-tuning GPT-2 for text generation
4. **Named Entity Recognition** - Fine-tuning BERT for NER tasks

## 📁 Files

- `sentiment_finetuning.py` - Traditional fine-tuning for sentiment analysis
- `lora_finetuning.py` - LoRA-based fine-tuning examples
- `text_generation_lora.py` - Text generation with LoRA
- `ner_finetuning.py` - Named Entity Recognition fine-tuning

## 🚀 Quick Start

### 1. Basic Sentiment Analysis Fine-tuning

```bash
python sentiment_finetuning.py
```

This will:
- Load a pretrained DistilBERT model
- Fine-tune on sentiment data
- Evaluate performance
- Save the trained model

### 2. LoRA Fine-tuning (Memory Efficient)

```bash
python lora_finetuning.py
```

Benefits of LoRA:
- 90%+ reduction in trainable parameters
- Faster training
- Lower memory usage
- Easy to switch between adapters

### 3. Text Generation with LoRA

```bash
python text_generation_lora.py
```

Fine-tune GPT-2 for domain-specific text generation.

### 4. Named Entity Recognition

```bash
python ner_finetuning.py
```

Fine-tune BERT for custom NER tasks.

## ⚙️ Configuration

Each script accepts command-line arguments:

```bash
python sentiment_finetuning.py \
    --model-name distilbert-base-uncased \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --output-dir ./results
```

## 📊 Expected Results

### Sentiment Analysis
- Accuracy: ~85-90%
- Training time: ~5-10 minutes (CPU)

### LoRA Fine-tuning
- Parameter reduction: 90-95%
- Similar performance to full fine-tuning
- Training time: 50% faster

### Text Generation
- Perplexity: ~30-40 (after fine-tuning)
- Coherent domain-specific text

### NER
- F1 Score: ~80-85%
- Token-level accuracy: ~90%

## 💡 Tips

1. **Start small**: Begin with fewer epochs and smaller datasets
2. **Monitor loss**: Watch for overfitting
3. **Use LoRA**: For large models, always prefer PEFT methods
4. **Save checkpoints**: Save intermediate models during training
5. **Evaluate regularly**: Check validation metrics after each epoch

## 🔧 Troubleshooting

**Out of Memory:**
```bash
# Reduce batch size
python sentiment_finetuning.py --batch-size 4

# Use LoRA instead
python lora_finetuning.py
```

**Slow Training:**
```bash
# Use smaller model
python sentiment_finetuning.py --model-name distilbert-base-uncased

# Enable mixed precision (if GPU available)
python sentiment_finetuning.py --fp16
```

## 📖 Additional Resources

- [Hugging Face Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 🎯 Assignment

Fine-tune a model for a custom task:
1. Choose a task (classification, NER, generation)
2. Prepare your dataset (at least 100 examples)
3. Fine-tune using both traditional and LoRA methods
4. Compare results and training time
5. Document your findings

**Deliverables:**
- Training scripts
- Trained model or adapter weights
- Evaluation metrics
- Comparison report
