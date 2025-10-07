"""
Calibrate thresholds for fine-tuned model.
This script finds optimal thresholds by analyzing the probability distribution
of the fine-tuned model on your validation data.

Usage:
python calibrate_finetuned_model.py \
    --ads ads_100_for_hyperparameters.jsonl \
    --non_ads non_ads_1000_for_hyperparameters.jsonl \
    --model_dir ./ovr_promo_ft \
    --output calibration_results.json
"""

import argparse
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from pathlib import Path


def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def normalize_text(text):
    """Basic text normalization."""
    import re
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"[_`~^]+", " ", text)
    text = re.sub(r"\s+([.,:;!?()])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def get_predictions(model, tokenizer, texts, device, batch_size=16):
    """Get model predictions for texts."""
    model.eval()
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        all_probs.append(probs.cpu().numpy())
    
    return np.vstack(all_probs)


def find_optimal_thresholds(y_true, y_probs):
    """Find optimal thresholds using different strategies."""
    results = {}
    
    # 1. F1-optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    
    results['f1_optimal'] = {
        'threshold': float(thresholds[optimal_idx]) if optimal_idx < len(thresholds) else 0.5,
        'f1': float(f1_scores[optimal_idx]),
        'precision': float(precision[optimal_idx]),
        'recall': float(recall[optimal_idx])
    }
    
    # 2. Balanced precision-recall (closest to equal)
    pr_diff = np.abs(precision - recall)
    balanced_idx = np.argmin(pr_diff)
    
    results['balanced'] = {
        'threshold': float(thresholds[balanced_idx]) if balanced_idx < len(thresholds) else 0.5,
        'f1': float(f1_scores[balanced_idx]),
        'precision': float(precision[balanced_idx]),
        'recall': float(recall[balanced_idx])
    }
    
    # 3. High precision threshold (95% precision)
    high_prec_idx = np.where(precision >= 0.95)[0]
    if len(high_prec_idx) > 0:
        high_prec_idx = high_prec_idx[0]
        results['high_precision'] = {
            'threshold': float(thresholds[high_prec_idx]) if high_prec_idx < len(thresholds) else 0.7,
            'f1': float(f1_scores[high_prec_idx]),
            'precision': float(precision[high_prec_idx]),
            'recall': float(recall[high_prec_idx])
        }
    
    # 4. High recall threshold (95% recall)
    high_rec_idx = np.where(recall >= 0.95)[0]
    if len(high_rec_idx) > 0:
        high_rec_idx = high_rec_idx[-1]  # Take the last one (highest threshold with 95% recall)
        results['high_recall'] = {
            'threshold': float(thresholds[high_rec_idx]) if high_rec_idx < len(thresholds) else 0.3,
            'f1': float(f1_scores[high_rec_idx]),
            'precision': float(precision[high_rec_idx]),
            'recall': float(recall[high_rec_idx])
        }
    
    # 5. Standard 0.5 threshold for reference
    pred_05 = (y_probs >= 0.5).astype(int)
    results['standard_0.5'] = {
        'threshold': 0.5,
        'f1': float(f1_score(y_true, pred_05)),
        'precision': float(precision_score(y_true, pred_05)),
        'recall': float(recall_score(y_true, pred_05))
    }
    
    return results, precision, recall, thresholds


def plot_calibration_curves(y_true, y_probs, results, output_path):
    """Create calibration visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    ax = axes[0, 0]
    ax.plot(recall, precision, 'b-', linewidth=2)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark optimal points
    for name, info in results.items():
        if name in ['f1_optimal', 'balanced']:
            ax.plot(info['recall'], info['precision'], 'ro', markersize=10)
            ax.annotate(f"{name}\n({info['threshold']:.3f})", 
                       xy=(info['recall'], info['precision']),
                       xytext=(10, -10), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # 2. F1 Score vs Threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    ax = axes[0, 1]
    ax.plot(thresholds, f1_scores[:-1], 'g-', linewidth=2)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score vs Threshold', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark optimal threshold
    opt_thresh = results['f1_optimal']['threshold']
    opt_f1 = results['f1_optimal']['f1']
    ax.axvline(opt_thresh, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax.plot(opt_thresh, opt_f1, 'ro', markersize=10)
    ax.annotate(f"Optimal: {opt_thresh:.3f}", xy=(opt_thresh, opt_f1),
               xytext=(10, 10), textcoords='offset points', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # 3. Probability distribution histogram
    ax = axes[1, 0]
    ax.hist(y_probs[y_true == 0], bins=50, alpha=0.6, label='Non-Ads', color='blue', density=True)
    ax.hist(y_probs[y_true == 1], bins=50, alpha=0.6, label='Ads', color='red', density=True)
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Probability Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Mark recommended thresholds
    for name, info in results.items():
        if name in ['f1_optimal', 'balanced']:
            ax.axvline(info['threshold'], linestyle='--', linewidth=2, alpha=0.7, 
                      label=f"{name}: {info['threshold']:.3f}")
    ax.legend(fontsize=9)
    
    # 4. Metrics comparison table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [['Strategy', 'Threshold', 'F1', 'Precision', 'Recall']]
    for name, info in results.items():
        table_data.append([
            name.replace('_', ' ').title(),
            f"{info['threshold']:.4f}",
            f"{info['f1']:.4f}",
            f"{info['precision']:.4f}",
            f"{info['recall']:.4f}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight optimal row
    for i in range(5):
        table[(1, i)].set_facecolor('#FFE082')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCalibration plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ads", required=True, help="Path to ads JSONL")
    parser.add_argument("--non_ads", required=True, help="Path to non-ads JSONL")
    parser.add_argument("--model_dir", default="./ovr_promo_ft", help="Fine-tuned model directory")
    parser.add_argument("--output", default="calibration_results.json", help="Output JSON file")
    parser.add_argument("--plot", default="calibration_plots.png", help="Output plot file")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    ads = load_jsonl(args.ads)
    non_ads = load_jsonl(args.non_ads)
    
    # Prepare texts and labels
    texts = [normalize_text(item.get('ft', '')) for item in ads + non_ads]
    y_true = np.array([1] * len(ads) + [0] * len(non_ads))
    
    print(f"Loaded {len(ads)} ads and {len(non_ads)} non-ads")
    
    # Load model
    print(f"\nLoading fine-tuned model from: {args.model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    
    # Get promotion label ID
    id2label = model.config.id2label
    promo_id = None
    for i, label in id2label.items():
        if label.lower() == "promotion":
            promo_id = i
            break
    
    if promo_id is None:
        raise ValueError("Could not find 'Promotion' label in model")
    
    print(f"Promotion label ID: {promo_id}")
    
    # Get predictions
    print("\nGetting model predictions...")
    all_probs = get_predictions(model, tokenizer, texts, device, args.batch_size)
    promo_probs = all_probs[:, promo_id]
    
    # Analyze probability distributions
    print("\nProbability distribution analysis:")
    print(f"  Ads - Min: {promo_probs[y_true==1].min():.4f}, "
          f"Mean: {promo_probs[y_true==1].mean():.4f}, "
          f"Median: {np.median(promo_probs[y_true==1]):.4f}, "
          f"Max: {promo_probs[y_true==1].max():.4f}")
    print(f"  Non-Ads - Min: {promo_probs[y_true==0].min():.4f}, "
          f"Mean: {promo_probs[y_true==0].mean():.4f}, "
          f"Median: {np.median(promo_probs[y_true==0]):.4f}, "
          f"Max: {promo_probs[y_true==0].max():.4f}")
    
    # Find optimal thresholds
    print("\nFinding optimal thresholds...")
    results, precision, recall, thresholds = find_optimal_thresholds(y_true, promo_probs)
    
    # Print results
    print("\n" + "="*70)
    print("CALIBRATION RESULTS")
    print("="*70)
    for name, info in results.items():
        print(f"\n{name.replace('_', ' ').upper()}:")
        print(f"  Threshold:  {info['threshold']:.6f}")
        print(f"  F1 Score:   {info['f1']:.4f}")
        print(f"  Precision:  {info['precision']:.4f}")
        print(f"  Recall:     {info['recall']:.4f}")
    
    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    opt = results['f1_optimal']
    print(f"Use threshold: {opt['threshold']:.6f}")
    print(f"Expected F1: {opt['f1']:.4f}, Precision: {opt['precision']:.4f}, Recall: {opt['recall']:.4f}")
    print(f"\nUpdate your model_approach.py with:")
    print(f"  --ad_threshold {opt['threshold']:.6f}")
    
    # Save results
    output_data = {
        'model_dir': args.model_dir,
        'n_ads': len(ads),
        'n_non_ads': len(non_ads),
        'promo_label_id': promo_id,
        'probability_stats': {
            'ads': {
                'min': float(promo_probs[y_true==1].min()),
                'mean': float(promo_probs[y_true==1].mean()),
                'median': float(np.median(promo_probs[y_true==1])),
                'max': float(promo_probs[y_true==1].max()),
            },
            'non_ads': {
                'min': float(promo_probs[y_true==0].min()),
                'mean': float(promo_probs[y_true==0].mean()),
                'median': float(np.median(promo_probs[y_true==0])),
                'max': float(promo_probs[y_true==0].max()),
            }
        },
        'thresholds': results,
        'recommended_threshold': opt['threshold']
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Create plots
    plot_calibration_curves(y_true, promo_probs, results, args.plot)
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()