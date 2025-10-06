"""
Evaluation script for the ad classification model.

Usage:
python evaluate_model.py --true_ads ads_1000.jsonl --true_non_ads non_ads_1000.jsonl --output_csv results.csv

This script will:
1. Combine true ads and non-ads into a test dataset
2. Run model_approach.py with best parameters from best_params.json
3. Calculate and display accuracy metrics and confusion matrix
4. Create a CSV file with detailed results (excluding correctly classified ads)
"""

import argparse
import json
import tempfile
import subprocess
import os
import csv
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

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
        '--model', best_params.get('model', 'classla/xlm-roberta-base-multilingual-text-genre-classifier'),
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
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
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

def create_detailed_csv(results, original_data, ground_truth, output_csv):
    """Create detailed CSV with predictions and analysis, excluding correctly classified ads."""
    print(f"Creating detailed CSV: {output_csv}")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        # Get all fields that actually exist in the original data (input JSONL)
        all_original_fields = set()
        for item in original_data:
            all_original_fields.update(item.keys())
        
        # Remove internal fields we don't want
        original_fields = sorted([f for f in all_original_fields if f not in ['ground_truth', 'type']])
        
        # Add the analysis columns you requested
        analysis_fields = ['predicted_label', 'classification_result', 'correct', 'article_link']
        
        # Final fieldnames: only original fields + the analysis fields
        fieldnames = original_fields + analysis_fields
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Count different types of results
        counts = {'correct_ad': 0, 'correct_non_ad': 0, 'missed_ad': 0, 'false_ad': 0}
        
        for i, (original, result, actual_is_ad) in enumerate(zip(original_data, results, ground_truth)):
            predicted_is_ad = result.get('is_ad_pred', False)
            
            # Determine if prediction is correct
            is_correct = (actual_is_ad == predicted_is_ad)
            
            # Determine classification result
            if actual_is_ad == predicted_is_ad:
                if actual_is_ad:
                    classification_result = 'correct_ad'  # True Positive
                    counts['correct_ad'] += 1
                    # Skip correctly classified ads - don't save to CSV
                    continue
                else:
                    classification_result = 'correct_non_ad'  # True Negative
                    counts['correct_non_ad'] += 1
            else:
                if actual_is_ad and not predicted_is_ad:
                    classification_result = 'missed_ad'  # False Negative
                    counts['missed_ad'] += 1
                else:
                    classification_result = 'false_ad'  # False Positive
                    counts['false_ad'] += 1
            
            # Create row with only original fields + analysis fields
            row = {}
            
            # Copy original data (excluding internal fields)
            for key, value in original.items():
                if key in fieldnames:
                    row[key] = value
            
            # Add the analysis columns
            row['predicted_label'] = 'ad' if predicted_is_ad else 'non-ad'
            row['classification_result'] = classification_result
            row['correct'] = is_correct
            
            # Create article link from ID
            article_id = original.get('id', '')
            row['article_link'] = f"https://impresso-project.ch/app/article/{article_id}" if article_id else ''
            
            # Ensure all fieldnames have values (empty string for missing)
            for field in fieldnames:
                if field not in row:
                    row[field] = ''
            
            writer.writerow(row)
        
        print(f"\nClassification results:")
        print(f"  Correct ads (TP): {counts['correct_ad']} (excluded from CSV)")
        print(f"  Correct non-ads (TN): {counts['correct_non_ad']}")
        print(f"  Missed ads (FN): {counts['missed_ad']}")
        print(f"  False ads (FP): {counts['false_ad']}")
        print(f"  Total saved to CSV: {counts['correct_non_ad'] + counts['missed_ad'] + counts['false_ad']}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate model with best parameters and create detailed analysis')
    parser.add_argument('--true_ads', required=True, help='JSONL file with true ads')
    parser.add_argument('--true_non_ads', required=True, help='JSONL file with true non-ads')
    parser.add_argument('--output_csv', required=True, help='CSV file for detailed results')
    parser.add_argument('--best_params', default='best_params.json', help='JSON file with best parameters')
    
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
        
        # Create detailed CSV
        create_detailed_csv(results, original_data, ground_truth, args.output_csv)
        print(f"\nDetailed results saved to: {args.output_csv}")
        
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
