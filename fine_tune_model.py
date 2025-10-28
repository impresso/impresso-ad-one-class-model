"""
Fine-tune xlm-roberta-base-multilingual-text-genre-classifier for ad detection.
Uses One-vs-Rest approach: enhances the Promotion class detection while preserving
the model's multi-class genre classification capabilities.

Usage:
python fine_tune_model.py \
    --ads ads_3800_finetuning.jsonl \
    --non_ads non_ads_3800_finetuning.jsonl \
    --output_dir ./fine_tuned_xlm \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5

Requirements:
pip install transformers datasets torch scikit-learn
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def prepare_ovr_dataset(ads_path, non_ads_path, tokenizer, test_size=0.15, seed=42):
    """
    Prepare dataset for One-vs-Rest fine-tuning.
    Maps: ad -> Promotion (label varies by model), non-ad -> aggregated other classes
    """
    # Load data
    ads = load_jsonl(ads_path)
    non_ads = load_jsonl(non_ads_path)
    
    print(f"Loaded {len(ads)} ads and {len(non_ads)} non-ads")
    
    # Get label mapping from the base model
    model_name = "classla/xlm-roberta-base-multilingual-text-genre-classifier"
    temp_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    id2label = temp_model.config.id2label
    label2id = {v: k for k, v in id2label.items()}
    
    # Find Promotion label ID
    promo_id = None
    for i, lab in id2label.items():
        if lab.lower() == "promotion":
            promo_id = i
            break
    
    if promo_id is None:
        raise ValueError("Could not find 'Promotion' label in model")
    
    print(f"Model labels: {id2label}")
    print(f"Promotion label ID: {promo_id}")
    
    # Prepare examples
    examples = []
    
    # Add ads - all map to Promotion
    for item in ads:
        text = item.get('ft', '')
        if text.strip():
            examples.append({
                'text': text,
                'label': promo_id,
                'original_type': 'ad',
                'language': item.get('lg', 'unknown')
            })
    
    # Add non-ads - distribute across other labels
    # Strategy: Use a weighted distribution favoring common non-promotion classes
    other_labels = [i for i in id2label.keys() if i != promo_id]
    
    # Weight distribution (favor News, Information, etc. for non-ads)
    label_weights = {}
    for label_id in other_labels:
        label_name = id2label[label_id].lower()
        if any(x in label_name for x in ['news', 'information', 'article']):
            label_weights[label_id] = 3.0
        elif any(x in label_name for x in ['opinion', 'editorial', 'legal', 'official']):
            label_weights[label_id] = 2.0
        else:
            label_weights[label_id] = 1.0
    
    # Normalize weights
    total_weight = sum(label_weights.values())
    label_probs = [label_weights[i] / total_weight for i in other_labels]
    
    for item in non_ads:
        text = item.get('ft', '')
        if text.strip():
            # Assign a non-promotion label based on weighted distribution
            assigned_label = np.random.choice(other_labels, p=label_probs)
            examples.append({
                'text': text,
                'label': int(assigned_label),
                'original_type': 'non-ad',
                'language': item.get('lg', 'unknown')
            })
    
    # Shuffle
    random.seed(seed)
    random.shuffle(examples)
    
    # Split train/val
    train_examples, val_examples = train_test_split(
        examples, test_size=test_size, random_state=seed, stratify=[e['original_type'] for e in examples]
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_examples)} ({sum(1 for e in train_examples if e['original_type'] == 'ad')} ads)")
    print(f"  Val: {len(val_examples)} ({sum(1 for e in val_examples if e['original_type'] == 'ad')} ads)")
    
    # Print label distribution
    print("\nTrain label distribution:")
    train_label_counts = Counter(e['label'] for e in train_examples)
    for label_id, count in sorted(train_label_counts.items()):
        print(f"  {id2label[label_id]}: {count}")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_dict({
        'text': [e['text'] for e in train_examples],
        'label': [e['label'] for e in train_examples],
        'original_type': [e['original_type'] for e in train_examples]
    })
    
    val_dataset = Dataset.from_dict({
        'text': [e['text'] for e in val_examples],
        'label': [e['label'] for e in val_examples],
        'original_type': [e['original_type'] for e in val_examples]
    })
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Remove text column (keep original_type for evaluation)
    train_dataset = train_dataset.remove_columns(['text'])
    val_dataset = val_dataset.remove_columns(['text'])
    
    return train_dataset, val_dataset, id2label, promo_id


def compute_metrics(eval_pred, promo_id):
    """
    Compute metrics focusing on binary ad/non-ad classification
    even though the model predicts multiple classes.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Convert to binary: is it Promotion or not?
    binary_preds = (predictions == promo_id).astype(int)
    binary_labels = (labels == promo_id).astype(int)
    
    # Multi-class accuracy
    accuracy = (predictions == labels).mean()
    
    # Binary metrics for ad detection
    tp = ((binary_preds == 1) & (binary_labels == 1)).sum()
    fp = ((binary_preds == 1) & (binary_labels == 0)).sum()
    tn = ((binary_preds == 0) & (binary_labels == 0)).sum()
    fn = ((binary_preds == 0) & (binary_labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    binary_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'binary_accuracy': binary_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XLM-RoBERTa genre classifier for ad detection")
    parser.add_argument("--ads", required=True, help="Path to ads JSONL file")
    parser.add_argument("--non_ads", required=True, help="Path to non-ads JSONL file")
    parser.add_argument("--output_dir", default="./ovr_promo_ft", help="Output directory for fine-tuned model")
    parser.add_argument("--model", default="classla/xlm-roberta-base-multilingual-text-genre-classifier")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint frequency")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load tokenizer and model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Determine device - MPS support
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model = model.to(device)
    
    # Prepare dataset
    print("\nPreparing dataset...")
    train_dataset, val_dataset, id2label, promo_id = prepare_ovr_dataset(
        args.ads, args.non_ads, tokenizer, seed=args.seed
    )
    
    # Training arguments (compatible with different transformers versions)
    training_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size * 2,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_dir": f"{args.output_dir}/logs",
        "logging_steps": 50,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "seed": args.seed,
        "report_to": "none",
        "push_to_hub": False,
        "use_cpu": False,  # Don't force CPU
    }
    
    # Add evaluation and save strategy (parameter names vary by version)
    try:
        # Try newer parameter names first
        training_args_dict["eval_strategy"] = "steps"
        training_args_dict["eval_steps"] = args.eval_steps
        training_args_dict["save_strategy"] = "steps"
        training_args_dict["save_steps"] = args.save_steps
        training_args = TrainingArguments(**training_args_dict)
    except TypeError:
        # Fall back to older parameter names
        training_args_dict.pop("eval_strategy", None)
        training_args_dict["evaluation_strategy"] = "steps"
        training_args_dict["eval_steps"] = args.eval_steps
        training_args_dict["save_strategy"] = "steps"
        training_args_dict["save_steps"] = args.save_steps
        training_args = TrainingArguments(**training_args_dict)
    
    # Only add fp16 for CUDA (not MPS or CPU)
    if torch.cuda.is_available():
        training_args.fp16 = True
    
    # Create trainer with custom metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, promo_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    
    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    
    # Save final model
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training info
    with open(f"{args.output_dir}/training_info.json", "w") as f:
        json.dump({
            "model_name": args.model,
            "promo_label_id": promo_id,
            "id2label": id2label,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        }, f, indent=2)
    
    # Final evaluation
    print("\nFinal evaluation:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Detailed evaluation on validation set
    print("\nDetailed validation set evaluation:")
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    # Binary classification report
    binary_preds = (pred_labels == promo_id).astype(int)
    binary_true = (true_labels == promo_id).astype(int)
    
    print("\nBinary Classification (Ad vs Non-Ad):")
    print(classification_report(binary_true, binary_preds, target_names=['Non-Ad', 'Ad']))
    
    print("\nConfusion Matrix (Ad Detection):")
    cm = confusion_matrix(binary_true, binary_preds)
    print(f"              Predicted Non-Ad  Predicted Ad")
    print(f"Actual Non-Ad       {cm[0,0]:6d}          {cm[0,1]:6d}")
    print(f"Actual Ad           {cm[1,0]:6d}          {cm[1,1]:6d}")
    
    print(f"\nModel saved to: {args.output_dir}")
    print("You can now use it in model_approach.py by changing the model loading line to:")
    print(f"  model = AutoModelForSequenceClassification.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()