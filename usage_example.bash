#!/bin/bash

# ==========================================================
# 1) Prepare training data as JSON
# ==========================================================
#    positives.json : ONLY advertisements (any language)
#    unlabeled.json : mixed corpus chunks (ads + non-ads)
#
# Supported JSON formats:
#   (A) Mapping: {"article_1": "text ...", "article_2": "text ..."}
#   (B) Array of objects: [{"id": "...", "text": "..."}, ...]
#   (C) JSONL: one JSON object per line

# ==========================================================
# 2) Train with defaults
# ==========================================================
# This will:
#   - Train a two-stage classifier:
#       Stage A: TF-IDF + Linear filter (high recall, ~30% keep)
#       Stage B: PU learning on multilingual embeddings
#   - Save models and reports under outputs_ads_json/

python train_ads_classifier_json.py \
  --pos_json data/positives.json \
  --u_json data/unlabeled.json \
  --out_dir outputs_ads_json \
  --embed_model intfloat/multilingual-e5-small

# ==========================================================
# 3) (Optional) Train with held-out validation sets
# ==========================================================
python train_ads_classifier_json.py \
  --pos_json data/pos_train.json \
  --u_json data/unlabeled_all.json \
  --val_pos_json data/pos_val.json \
  --val_u_json data/u_val.json \
  --out_dir outputs_ads_json

# ==========================================================
# 4) (Optional) Use a custom field name for text
# ==========================================================
python train_ads_classifier_json.py \
  --pos_json data/positives_array.json \
  --u_json data/unlabeled_array.json \
  --text_key content \
  --out_dir outputs_ads_json

# ==========================================================
# 5) Inference: classify NEW articles after training
# ==========================================================
# After training, you no longer need unlabeled.json.
# Use the trained models to classify any new JSON file.
# The script outputs a JSON with probabilities + labels.

python predict_json.py \
  --input_json data/new_articles.json \
  --model_dir outputs_ads_json \
  --output_json predictions.json

# Example output (predictions.json):
# {
#   "article_001": {"score": 0.98, "label": "advertisement"},
#   "article_002": {"score": 0.12, "label": "not_advertisement"},
#   "article_003": {"score": 0.87, "label": "advertisement"}
# }
