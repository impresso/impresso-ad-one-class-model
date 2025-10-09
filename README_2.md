# Ad Classification Model - Setup Guide

## 📊 Data Preparation

### Required Datasets

You'll need to prepare three separate datasets for both ads and non-ads:

| Dataset | Recommended Size | Purpose |
|---------|-----------------|---------|
| Fine-tuning samples | 3,800 | Training the model |
| Hyperparameter optimization | 100 | Tuning model parameters |
| Evaluation set | 100 | Testing accuracy (untouched) |

### Data Format

All data should be in `.jsonl` format with one JSON object per line:

**Ads:**
```json
{"id": "ARTICLE_ID", "lg": "fr", "ft": "ARTICLE_TEXT", "type": "ad"}
```

**Non-ads:**
```json
{"id": "ARTICLE_ID", "lg": "fr", "ft": "ARTICLE_TEXT", "type": "non-ad"}
```

### Collection Process

**For ads:** Sample ad IDs, retrieve their text, and add the `"type"` field.

**For non-ads:** Since true non-ads aren't explicitly labeled, you'll need to:
1. Sample articles with topics typical for non-ads
2. Manually annotate them for accuracy

💡 *The `helper_scripts_for_data_preparation` folder contains utilities for this process. Feel free to reach out if you need clarification.*

### Recommended Approach

- Use topic modeling to select the bulk of non-ads (e.g., 3,800 for fine-tuning)
- Manually annotate smaller sets for hyperparameter tuning and evaluation (200 total) to ensure highest accuracy
- Consider using ChatGPT with article screenshots for efficient manual annotation

---

## 🔍 Classification Methodology

This classifier uses a **hybrid approach** combining:

- **RoBERTa model** for semantic understanding
- **Algorithmic rules** for pattern detection (e.g., presence of phone numbers increases ad likelihood)

The model is fine-tuned using an "ads vs. rest" approach, adapting the default 9-class classifier to better understand your historical data.

📖 **Detailed documentation:**
- Classification approach: `ad-classification-doc.html`
- Fine-tuning details: `finetuning_doc.html`

---

## 🚀 Usage Instructions

### Step 1: Fine-tune the Model

```bash
python fine_tune_xgenre.py \
  --ads ads_3800_finetuning.jsonl \
  --non_ads non_ads_3800_finetuning.jsonl \
  --output_dir ./fine_tuned_xlm \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5
```

**Note:** Save the model to your specified `output_dir` - you'll need this path for the next steps.

### Step 2: Update Model Path

After fine-tuning, update the model path in `model_approach.py` to point to your newly trained model.

### Step 3: Optimize Hyperparameters

Find the optimal hyperparameters for your fine-tuned model:

```bash
python optimize_hyperparams.py \
  --ads ads_100_for_hyperparameters.jsonl \
  --non_ads non_ads_100_for_hyperparameters.jsonl \
  --output best_params.json \
  --max_configs 120
```

This will test up to 120 different configurations and save the best parameters.

### Step 4: Evaluate Model Performance

Test your model on the untouched evaluation set:

```bash
python evaluate_model.py \
  --true_ads ads_100_for_testing.jsonl \
  --true_non_ads non_ads_100_for_testing.jsonl \
  --output_csv results.csv
```

The script automatically uses the optimized parameters from `best_params.json`.

---

## 📝 Summary

1. **Prepare** your datasets (fine-tuning, hyperparameter tuning, evaluation)
2. **Fine-tune** the model with your data
3. **Optimize** hyperparameters for best performance
4. **Evaluate** on the test set to measure accuracy

For questions or additional support, please don't hesitate to reach out!