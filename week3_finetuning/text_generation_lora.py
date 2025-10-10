"""
Text Generation with LoRA
Fine-tune GPT-2 for domain-specific text generation using LoRA.

Usage:
    python text_generation_lora.py --epochs 3 --prompt "Machine learning is"
"""

import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_training_data():
    """Create sample training data for text generation."""
    texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to process information.",
        "Natural language processing helps computers understand and generate human language.",
        "Transformers revolutionized NLP with their attention mechanisms.",
        "BERT is a bidirectional transformer model that excels at understanding context.",
        "GPT models are autoregressive and excel at text generation tasks.",
        "Fine-tuning adapts pretrained models to specific tasks with minimal data.",
        "Transfer learning leverages knowledge from one task to improve performance on another.",
        "Neural networks learn patterns through backpropagation and gradient descent.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
    ] * 10  # Repeat for more training data

    return Dataset.from_dict({"text": texts})


def main(args):
    """Main training function."""
    logger.info("Starting text generation fine-tuning with LoRA...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Configure LoRA
    logger.info(f"Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn", "c_proj"],  # GPT-2 specific
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = create_training_data()

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        weight_decay=args.weight_decay,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save adapter
    logger.info(f"Saving LoRA adapter to {args.output_dir}/text_gen_adapter")
    model.save_pretrained(f"{args.output_dir}/text_gen_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/text_gen_adapter")

    # Generate text
    logger.info("\nGenerating text samples...")
    model.eval()

    prompts = [
        "Machine learning is",
        "Deep learning",
        "Natural language processing",
        args.prompt if args.prompt else "Artificial intelligence"
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=args.max_gen_length,
                num_return_sequences=1,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"\nPrompt: {prompt}")
        logger.info(f"Generated: {generated_text}")

    logger.info("\nText generation fine-tuning complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 with LoRA for text generation")

    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--output-dir", type=str, default="./text_gen_results")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # LoRA parameters
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    # Generation parameters
    parser.add_argument("--prompt", type=str, default="Artificial intelligence")
    parser.add_argument("--max-gen-length", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()
    main(args)
