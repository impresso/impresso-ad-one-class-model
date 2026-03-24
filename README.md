# Fine-Tuning Handover

This repo fine-tunes `classla/xlm-roberta-base-multilingual-text-genre-classifier` for ad vs non-ad detection.

## Data Split

Keep the datasets separate and do not reuse examples across splits:

- `ads_3800_finetuning.jsonl` + `non_ads_3800_finetuning.jsonl`: training data
- `ads_100_for_hyperparameters.jsonl` + `non_ads_100_for_hyperparameters.jsonl`: hyperparameter tuning
- `ads_100_for_testing.jsonl` + `non_ads_100_for_testing.jsonl`: final test set

Recommended ratio in this repo:

- 3800 ads + 3800 non-ads for fine-tuning
- 100 ads + 100 non-ads for hyperparameter search
- 100 ads + 100 non-ads for final evaluation

## JSONL Format

Each line must be one JSON object.

Minimum fields for all scripts:

- `ft`: full OCR/text string to classify

Recommended fields:

- `id`: article id, used in reports and CSV output
- `lg`: language code such as `fr`, `de`, `lb`; can be `null`
- `type`: `ad` or `non-ad`; mainly useful for mixed files and bookkeeping

Example:

```json
{"id":"IMP-1978-10-12-a-i0288","lg":"fr","type":"ad","ft":"A LOUER ..."}
```

## Workflow

### 1. Fine-tune the model

Call:

```bash
python fine_tune_model.py \
  --ads ads_3800_finetuning.jsonl \
  --non_ads non_ads_3800_finetuning.jsonl \
  --model classla/xlm-roberta-base-multilingual-text-genre-classifier \
  --output_dir fine_tuned_xlm \
  --epochs 4 \
  --batch_size 8 \
  --learning_rate 1.5e-5
```

What it does:

- trains the model
- internally splits the finetuning files into train/validation
- saves the model to `fine_tuned_xlm/`

Output:

- model directory with checkpoints
- `training_info.json`

### 2. Optimize inference parameters

Call:

```bash
python optimize_hyperparams.py \
  --ads ads_100_for_hyperparameters.jsonl \
  --non_ads non_ads_100_for_hyperparameters.jsonl \
  --model fine_tuned_xlm \
  --output best_params.json
```

What it does:

- finds the best inference settings for the trained model
- tunes threshold, pooling, chunking, temperature, and related options

Output:

- `best_params.json`: best parameters to use for inference
- `optimization_log.json`: search log

### 3. Optional: simple threshold calibration

Call:

```bash
python calibrate_finetuned_model.py \
  --ads ads_100_for_hyperparameters.jsonl \
  --non_ads non_ads_100_for_hyperparameters.jsonl \
  --model_dir fine_tuned_xlm
```

What it does:

- finds a good raw `Promotion` threshold only
- simpler than `optimize_hyperparams.py`

Output:

- `calibration_results.json`
- `calibration_plots.png`

Use this mainly as a quick check. If you already use `optimize_hyperparams.py`, this step is mostly redundant.

### 4. Final evaluation

Call:

```bash
python evaluate_model.py \
  --true_ads ads_100_for_testing.jsonl \
  --true_non_ads non_ads_100_for_testing.jsonl \
  --best_params best_params.json \
  --output_csv results.csv
```

What it does:

- runs the model with the best parameters
- evaluates on the held-out test set

Output:

- terminal metrics
- `results.csv` with detailed examples

## Important Caveat

`model_approach.py` currently loads `./fine_tuned_xlm` directly. If you save a new model directory, update that path or keep using the name `fine_tuned_xlm`.
