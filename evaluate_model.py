"""
Evaluation script for the ad classification model.

Usage:
python evaluate_model.py 
    --true_ads cross-validation-cleaning/ads_fold_1_testing.jsonl \
    --true_non_ads cross-validation-cleaning/non_ads_fold_1_testing.jsonl \
    --ads_predictions cross-validation-cleaning/predictions/ads-fold-1-predictions.jsonl \
    --non_ads_predictions cross-validation-cleaning/predictions/non-ads-fold-1-predictions.jsonl \
    --best_params cross-validation-cleaning/best_params/best_params_fold_1.json \
    --model fine_tuned_xlm_fold_1


Test:
python evaluate_model.py --true_ads ads_100_for_testing.jsonl --true_non_ads non_ads_100_for_testing.jsonl --ads_predictions cross-validation-cleaning/ads-fold-1.jsonl --non_ads_predictions cross-validation-cleaning/non-ads-fold-1.jsonl --best_params cross-validation-cleaning/best_params/best_params_fold_1.json


This script will:
1. Combine true ads and non-ads into a test dataset
2. Run model_approach.py with best parameters from best_params.json
3. Calculate and display accuracy metrics and confusion matrix
4. Create a JSONL file with detailed results (excluding correctly classified ads)
"""

import argparse
import json
import tempfile
import subprocess
import os
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_best_params(best_params_file="best_params.json"):
    """Load best parameters from optimization results."""
    with open(best_params_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['best_params']

def create_combined_test_file(true_ads_file, true_non_ads_file):
    """Create combined test file with ground truth labels."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
    
    # Load true ads
    with open(true_ads_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item['ground_truth'] = True  # True = ad
                # Remove ground_truth for model input version
                model_item = {k: v for k, v in item.items() if k != 'ground_truth'}
                temp_file.write(json.dumps(model_item, ensure_ascii=False) + '\n')
    
    # Load true non-ads
    with open(true_non_ads_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item['ground_truth'] = False  # False = non-ad
                # Remove ground_truth for model input version
                model_item = {k: v for k, v in item.items() if k != 'ground_truth'}
                temp_file.write(json.dumps(model_item, ensure_ascii=False) + '\n')
    
    temp_file.close()
    return temp_file.name

def run_model_with_best_params(input_file, best_params):
    """Run model_approach.py with best parameters."""
    output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    output_file.close()
    
    cmd = [
        'python', 'model_approach.py',
        '--in', input_file,
        '--out', output_file.name,
        '--model', best_params.get('model'),
        '--batch_size', str(best_params.get('batch_size', 16)),
        '--max_length', str(best_params.get('max_length', 512)),
        '--chunk_words', str(best_params.get('chunk_words', 0)),
        '--pool', best_params.get('pool', 'logits_weighted'),
        '--ad_threshold', str(best_params.get('ad_threshold', 0.60)),
        '--short_len', str(best_params.get('short_len', 40)),
        '--short_bonus', str(best_params.get('short_bonus', 0.20)),
        '--min_words', str(best_params.get('min_words', 0)),
        '--temperature', str(best_params.get('temperature', 1.0))
    ]
    
    if best_params.get('lang_thresholds'):
        cmd.extend(['--lang_thresholds', best_params['lang_thresholds']])
    
    if best_params.get('meta_clf'):
        cmd.extend(['--meta_clf', best_params['meta_clf']])
    
    try:
        print("Running model with best parameters...")
        print("Command:", ' '.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Error running model: {result.stderr}")
            return None
        return output_file.name
    except subprocess.TimeoutExpired:
        print("Model execution timed out")
        return None
    except Exception as e:
        print(f"Error running model: {e}")
        return None

def load_results(output_file):
    """Load model results."""
    results = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def load_ground_truth(true_ads_file, true_non_ads_file):
    """Load ground truth labels in the same order as the combined file."""
    ground_truth = []
    original_data = []
    
    # Load true ads
    with open(true_ads_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                ground_truth.append(True)
                original_data.append(item)
    
    # Load true non-ads
    with open(true_non_ads_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                ground_truth.append(False)
                original_data.append(item)
    
    return ground_truth, original_data

def process_labels(labels):
    return [1 if label == "ad" else 0 for label in labels]

def normalize_label(label):
    # Accepts int, str, or None and returns "ad" or "non-ad"
    if label in ["ad", 1, "1", True]:
        return "ad"
    elif label in ["non-ad", 0, "0", False]:
        return "non-ad"
    # fallback: treat anything else as non-ad (or raise error)
    return "non-ad"

def normalize_labels(labels):
    return [normalize_label(l) for l in labels]

def evaluate_results(results, ground_truth):
    """Calculate accuracy metrics."""
    # Get predictions
    y_pred = [result.get('is_ad_pred', False) for result in results]
    y_true = ground_truth
    
    if len(y_true) != len(y_pred):
        print(f"Warning: Ground truth length ({len(y_true)}) != predictions length ({len(y_pred)})")
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    
    # Normalize labels to "ad" and "non-ad"
    y_true = normalize_labels(y_true)
    y_pred = normalize_labels(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="ad")
    recall = recall_score(y_true, y_pred, pos_label="ad")
    f1 = f1_score(y_true, y_pred, pos_label="ad")
    cm = confusion_matrix(y_true, y_pred, labels=["non-ad", "ad"])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'n_samples': len(y_true),
        'y_true': y_true,
        'y_pred': y_pred
    }

def save_predictions_jsonl(original_data, predictions, output_path):
    """
    Save predictions to JSONL file with the same structure as input plus predicted_type key
    
    Args:
        original_data: List of original input dictionaries
        predictions: List of predicted labels
        output_path: Path to output JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for item, prediction in zip(original_data, predictions):
            # Create a copy of the original item and add predicted_type
            output_item = item.copy()
            output_item['predicted_type'] = 'ad' if prediction else 'non-ad'
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
    
    print(f"Predictions saved to {output_path}")

def save_predictions_by_type(original_data, predictions, ads_output_path, non_ads_output_path):
    """
    Save predictions to separate JSONL files based on original type label
    
    Args:
        original_data: List of original input dictionaries
        predictions: List of predicted labels (True/False)
        ads_output_path: Path to output JSONL file for ads
        non_ads_output_path: Path to output JSONL file for non-ads
    """
    with open(ads_output_path, 'w', encoding='utf-8') as ads_file, \
         open(non_ads_output_path, 'w', encoding='utf-8') as non_ads_file:
        
        for item, prediction in zip(original_data, predictions):
            # Create a copy of the original item and add predicted_type
            output_item = item.copy()
            output_item['predicted_type'] = 'ad' if prediction else 'non-ad'
            
            # Write to appropriate file based on original type
            if item.get('type') == 'ad':
                ads_file.write(json.dumps(output_item, ensure_ascii=False) + '\n')
            else:
                non_ads_file.write(json.dumps(output_item, ensure_ascii=False) + '\n')
    
    print(f"Ads predictions saved to {ads_output_path}")
    print(f"Non-ads predictions saved to {non_ads_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--true_ads', required=True, help='JSONL file with true ads')
    parser.add_argument('--true_non_ads', required=True, help='JSONL file with true non-ads')
    parser.add_argument('--ads_predictions', default='cross-validation-cleaning/ads_fold_1_predictions.jsonl', help='Ads predictions JSONL output file')
    parser.add_argument('--non_ads_predictions', default='cross-validation-cleaning/non_ads_fold_1_predictions.jsonl', help='Non-ads predictions JSONL output file')
    # parser.add_argument('--output_csv', default='predictions.csv', help='CSV output file for predictions')
    parser.add_argument('--best_params', default='best_params.json', help='JSON file with best parameters')
    parser.add_argument('--model', default='fine_tuned_xlm_fold_1', help='HF model id')

    args = parser.parse_args()
    
    # Verify input files exist
    if not os.path.exists(args.true_ads):
        print(f"Error: True ads file not found: {args.true_ads}")
        sys.exit(1)
    if not os.path.exists(args.true_non_ads):
        print(f"Error: True non-ads file not found: {args.true_non_ads}")
        sys.exit(1)
    if not os.path.exists(args.best_params):
        print(f"Error: Best parameters file not found: {args.best_params}")
        sys.exit(1)
    
    # Load best parameters
    print("Loading best parameters...")
    try:
        best_params = load_best_params(args.best_params)
        print("Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print()
    except Exception as e:
        print(f"Error loading best parameters: {e}")
        sys.exit(1)
    
    # Create combined test file
    print("Creating combined test dataset...")
    input_file = create_combined_test_file(args.true_ads, args.true_non_ads)
    
    # Load ground truth
    ground_truth, original_data = load_ground_truth(args.true_ads, args.true_non_ads)
    print(f"Loaded {len(ground_truth)} samples ({sum(ground_truth)} ads, {len(ground_truth) - sum(ground_truth)} non-ads)")
    
    try:
        # Run model
        output_file = run_model_with_best_params(input_file, best_params)
        
        if output_file is None:
            print("Failed to run model")
            sys.exit(1)
        
        # Load results
        results = load_results(output_file)
        print(f"\nProcessed {len(results)} items")
        
        # Evaluate results
        metrics = evaluate_results(results, ground_truth)
        
        # Print metrics
        print("="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1']:.4f}")
        print(f"Samples:   {metrics['n_samples']}")
        print("\nConfusion Matrix:")
        print("                Predicted")
        print("                Non-Ad    Ad")
        print(f"Actual Non-Ad    {metrics['confusion_matrix'][0][0]:4d}   {metrics['confusion_matrix'][0][1]:4d}")
        print(f"       Ad        {metrics['confusion_matrix'][1][0]:4d}   {metrics['confusion_matrix'][1][1]:4d}")
        
        # Save predictions to JSONL files separated by original type
        save_predictions_by_type(
            original_data, 
            [result.get('is_ad_pred', False) for result in results], 
            args.ads_predictions, 
            args.non_ads_predictions
        )
        
        print(f"\nDetailed results saved to:")
        print(f"  Ads: {args.ads_predictions}")
        print(f"  Non-ads: {args.non_ads_predictions}")
        
        # Clean up temporary files
        os.unlink(input_file)
        os.unlink(output_file)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        # Clean up on error
        if os.path.exists(input_file):
            os.unlink(input_file)
        sys.exit(1)


if __name__ == "__main__":
    main()

