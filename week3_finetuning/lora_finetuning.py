"""
LoRA Fine-tuning
Parameter-efficient fine-tuning using Low-Rank Adaptation.

Usage:
    python lora_finetuning.py --epochs 5 --lora-r 8
"""

import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataset():
    """Create a sample dataset."""
    data = {
        "text": [
            "I love this product! It's amazing!",
            "This is the worst purchase I've ever made.",
            "Great quality and fast shipping!",
            "Terrible experience, very disappointed.",
            "Absolutely fantastic! Highly recommend!",
            "Not worth the money. Poor quality.",
            "Exceeded my expectations!",
            "Would not buy again. Total waste.",
            "Best product ever! So happy with it!",
            "Horrible quality, broke after one day.",
        ] * 10,
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10
    }

    dataset = Dataset.from_dict(data)
    return dataset.train_test_split(test_size=0.2, seed=42)


def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    return dataset.map(tokenize_function, batched=True)


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='weighted')
    }


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main(args):
    """Main training function."""
    logger.info("Starting LoRA fine-tuning...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )

    # Configure LoRA
    logger.info(f"Configuring LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_lin", "v_lin"] if "distilbert" in args.model_name.lower()
                       else ["query", "value"],
    )

    # Create PEFT model
    model = get_peft_model(base_model, lora_config)

    # Print parameter information
    total_params, trainable_params = count_parameters(model)
    logger.info(f"\nParameter Statistics:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    logger.info(f"  Parameter reduction: {100 * (1 - trainable_params / total_params):.2f}%")

    # Create and tokenize dataset
    logger.info("\nPreparing dataset...")
    dataset = create_dataset()
    tokenized_train = tokenize_dataset(dataset["train"], tokenizer)
    tokenized_test = tokenize_dataset(dataset["test"], tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("\nStarting training...")
    trainer.train()

    # Evaluate
    logger.info("\nEvaluating model...")
    results = trainer.evaluate()

    logger.info("\nEvaluation Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")

    # Save LoRA adapter
    logger.info(f"\nSaving LoRA adapter to {args.output_dir}/lora_adapter")
    model.save_pretrained(f"{args.output_dir}/lora_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/lora_adapter")

    # Test predictions
    logger.info("\nTesting predictions...")
    test_texts = [
        "This is amazing!",
        "Very disappointed.",
        "It's okay, nothing special."
    ]

    model.eval()
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            sentiment = "positive" if prediction == 1 else "negative"
            confidence = torch.softmax(outputs.logits, dim=-1)[0][prediction].item()

        logger.info(f"  Text: {text}")
        logger.info(f"  Prediction: {sentiment} (confidence: {confidence:.4f})")

    # Instructions for loading
    logger.info("\n" + "="*50)
    logger.info("To load this adapter later:")
    logger.info(f"  from peft import PeftModel")
    logger.info(f"  base_model = AutoModelForSequenceClassification.from_pretrained('{args.model_name}')")
    logger.info(f"  model = PeftModel.from_pretrained(base_model, '{args.output_dir}/lora_adapter')")
    logger.info("="*50)

    logger.info("\nLoRA fine-tuning complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for text classification")

    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=str, default="./lora_results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # LoRA specific arguments
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")

    args = parser.parse_args()
    main(args)
