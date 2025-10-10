"""
Sentiment Analysis Fine-tuning
Traditional full fine-tuning approach for text classification.

Usage:
    python sentiment_finetuning.py --epochs 3 --batch-size 8
"""

import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataset():
    """Create a sample sentiment dataset."""
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


def tokenize_dataset(dataset, tokenizer, max_length=128):
    """Tokenize the dataset."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    return dataset.map(tokenize_function, batched=True)


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')

    return {"accuracy": accuracy, "f1": f1}


def main(args):
    """Main training function."""
    logger.info("Starting sentiment analysis fine-tuning...")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )

    # Create dataset
    logger.info("Creating dataset...")
    dataset = create_dataset()
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    logger.info(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    # Tokenize
    logger.info("Tokenizing dataset...")
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=args.fp16 and device == "cuda",
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
    logger.info("Starting training...")
    trainer.train()

    # Evaluate
    logger.info("Evaluating model...")
    results = trainer.evaluate()

    logger.info("\nEvaluation Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")

    # Save model
    logger.info(f"Saving model to {args.output_dir}/final_model")
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")

    # Test predictions
    logger.info("\nTesting predictions...")
    test_texts = [
        "This is incredible! Best purchase ever!",
        "Disappointed with the quality.",
        "Neutral opinion about this item."
    ]

    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            sentiment = "positive" if prediction == 1 else "negative"
            confidence = torch.softmax(outputs.logits, dim=-1)[0][prediction].item()

        logger.info(f"  Text: {text}")
        logger.info(f"  Prediction: {sentiment} (confidence: {confidence:.4f})")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for sentiment analysis")

    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training"
    )

    args = parser.parse_args()
    main(args)
