# impresso ad classifier (one-class / ads-vs-rest)

Binary **AD / NOT_AD** classifier for historical newspaper content items (FR / DE / LB, plus others), built for the [impresso](https://impresso-project.ch) corpus.

It fine-tunes the 9-genre model [`classla/xlm-roberta-base-multilingual-text-genre-classifier`](https://huggingface.co/classla/xlm-roberta-base-multilingual-text-genre-classifier) so that its `Promotion` class becomes an "is this an ad?" detector. At inference, that probability goes through a tuned decision layer: chunking and pooling, temperature, per-language thresholds, a short-text bonus, and regex rules for prices, phone numbers and ad cue words.

**Published model:** [`impresso-project/impresso-ad-classification-xlm-one-class`](https://huggingface.co/impresso-project/impresso-ad-classification-xlm-one-class) on the Hugging Face Hub. Tags `v1.0` and `v2.0`; `main` = v2.

```python
from transformers import pipeline
pipe = pipeline("text-classification",
                model="impresso-project/impresso-ad-classification-xlm-one-class",
                trust_remote_code=True, revision="v2.0")
pipe("Appartement 3 pièces à louer, Fr. 2'100.–, tél. 079 ...")
```

---

## Model versions

| | **v1.0** (Nov 2025) | **v2.0** (May 2026, current) |
|---|---|---|
| Training data | 3725 ads + 3725 non-ads (v1 3800 set after CV cleaning) | 8623 ads + 8623 non-ads: [`fine-tuning-data/`](fine-tuning-data/) |
| Train / val split (internal, 85/15) | 6332 / 1118 | 14659 / 2587 |
| Training | 5 epochs, lr 2e-5, batch 16 | 5 epochs, lr 1.5e-5, batch 16 |
| Inference params | [`cross-validation-cleaning/best_params/best_params_final.json`](cross-validation-cleaning/best_params/best_params_final.json) | [`best_params.json`](best_params.json) |
| Tuning set | 200 + 200 | 300 + 300: [`fine-tuning-data/*_300_for_hyperparameters.jsonl`](fine-tuning-data/) |
| Macro-F1 on tuning set | 0.945 (P_ad 0.93, R_ad 0.965) | 0.912 (P_ad 0.95, R_ad 0.87) |
| How the data was built | [v1 data](#v1-data-sep-oct-2025) | [v2 data](#v2-data-mar-may-2026) |

The macro-F1 numbers are measured on *different* tuning sets, so they can't be compared directly. Both versions share the same held-out test set (`fine-tuning-data/*_100_for_testing.jsonl`, 100 + 100). That set is the fair place to compare them, but no v2 test-set result is recorded in the repo yet.

---

## How the classifier works

1. **Fine-tuning** ([`fine_tune_model.py`](fine_tune_model.py)): ads are labelled `Promotion` (id 8). Each non-ad gets a *random* non-Promotion genre, weighted towards News/Information (weight 3) and Opinion/Legal (weight 2) over the rest (weight 1). This sharpens `Promotion` while keeping the 9-class head ("one-vs-rest"). The other 8 genre outputs are therefore no longer meaningful genre predictions. 15% of the data is held out (stratified) for validation and early stopping.
2. **Inference** ([`model_approach.py`](model_approach.py) for CLI and research; [`pipeline.py`](pipeline.py) is the same logic packaged as a HF custom pipeline):
   - normalise text, optionally split into `chunk_words` chunks
   - pool chunk logits (`max`, `mean`, `logits_max`, `logits_mean`, `logits_weighted`), apply `temperature`
   - take P(`Promotion`) and compare it with a per-language threshold (`lang_thresholds`, fallback `ad_threshold`). Short texts (≤ `short_len` words) get `short_bonus` subtracted from the threshold.
   - the threshold and score are then nudged by rules. Rule hits (price, phone, ad cues, m², rooms, address, ZIP, year, km) lower the threshold. Signs of news, obituaries, official or legal text, or events raise the bar.
3. **Tuning** ([`optimize_hyperparams.py`](optimize_hyperparams.py)): caches logits once, searches up to `--max_configs` parameter combinations on the tuning set, re-validates the finalists by running `model_approach.py`, and writes `best_params.json` plus `optimization_log.json`.

Details: [docs/how-ad-classification-works.md](docs/how-ad-classification-works.md) (inference, rules, tuning) · [docs/how-the-model-is-fine-tuned.md](docs/how-the-model-is-fine-tuned.md) (training).

---

## Repository layout

```
fine_tune_model.py        1. fine-tune the genre model (ads -> Promotion)
optimize_hyperparams.py   2. search inference params on the tuning set -> best_params.json
model_approach.py            inference CLI used by 2 and 3 (JSONL in -> JSONL with is_ad_pred out)
evaluate_model.py         3. score the held-out test set using best_params.json
pipeline.py               HF custom pipeline (copy of what is on the Hub, v2 defaults)
model_pushing.py          4. upload a model dir + pipeline.py to the HF Hub
best_params.json          v2 inference params (includes "model": "fine_tuned_xlm_2")

fine-tuning-data/                    v2 train / tune / test splits (current)
data_preparation_second_finetuning/  how the v2 data was built (LLM + human relabelling)
cross-validation-cleaning/           v1 4-fold CV: scripts + per-fold best params (fold data is local only)
helper_scripts_for_data_preparation/ v1 data collection: S3 text fetch, cleaning, sampling
docs/                                how classification and fine-tuning work
cookbook/                            git submodule: impresso-make-cookbook (S3 helpers, Makefile includes)
```

Not in git (see `.gitignore`): trained model folders (`fine_tuned_xlm*/`, `fine_tuned_ad_classification_xlm/`), CV fold data, prediction dumps, `results.csv`, `optimization_log.json`. **Model weights only live on the HF Hub.** The local `fine_tuned_*` folders only hold configs and tokenizers.

---

## Setup

```bash
git clone --recurse-submodules git@github.com:impresso/impresso-ad-one-class-model.git
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp dotenv.sample .env   # S3 keys (data collection), OPENAI_API_KEY (v2 relabelling)
```

Training uses CUDA (fp16), Apple Silicon MPS, or CPU, whichever is available.

## Data format

One JSON object per line. Only `ft` is required for inference. For training and evaluation the class comes from *which file* a row is in, not from a field.

```json
{"id": "IMP-1978-10-12-a-i0288", "lg": "fr", "ft": "A LOUER ...", "type": "ad"}
```

`id` is the impresso content-item id (`<newspaper>-<YYYY-MM-DD>-<edition>-i<NNNN>`). `lg` may be `null`; with the v2 params anything that isn't `fr` uses `ad_threshold` (0.9612). Article link: `https://impresso-project.ch/app/article/<id>`.

Keep the splits disjoint. In the current data, train, tuning and test share 0 ids.

---

## Workflow (reproducing v2)

```bash
# 1. fine-tune -> ./fine_tuned_xlm_2
python fine_tune_model.py \
  --ads fine-tuning-data/ads_8623_finetuning.jsonl \
  --non_ads fine-tuning-data/non_ads_8623_finetuning.jsonl \
  --output_dir ./fine_tuned_xlm_2 --epochs 5 --batch_size 16 --learning_rate 1.5e-5

# 2. tune inference params -> best_params.json (+ optimization_log.json)
python optimize_hyperparams.py \
  --ads fine-tuning-data/ads_300_for_hyperparameters.jsonl \
  --non_ads fine-tuning-data/non_ads_300_for_hyperparameters.jsonl \
  --model fine_tuned_xlm_2 --output best_params.json --max_configs 120

# 3. evaluate on the untouched test set (the model is read from best_params["model"])
python evaluate_model.py \
  --true_ads fine-tuning-data/ads_100_for_testing.jsonl \
  --true_non_ads fine-tuning-data/non_ads_100_for_testing.jsonl \
  --best_params best_params.json

# ad-hoc inference on any JSONL
python model_approach.py --in input.jsonl --out out.jsonl --model fine_tuned_xlm_2 \
  --pool logits_mean --temperature 1.2 --chunk_words 200 \
  --ad_threshold 0.9612 --lang_thresholds "fr:0.8124,other:0.9570"

# 4. publish: edit LOCAL_DIR / REPO_ID in model_pushing.py, put the tuned params into
#    _DEFAULTS in pipeline.py, then:
python model_pushing.py   # prompts for an HF token; tag the release on the Hub afterwards
```

---

## Where the data came from

### v1 data (Sep–Oct 2025)

- **Ads**: content items already labelled as ads in impresso. IDs were sampled, then text was fetched from S3 with [`helper_scripts_for_data_preparation/s3_compiler.py`](helper_scripts_for_data_preparation/s3_compiler.py) and cleaned (length and OCR-quality filters) with [`clean_jsonl.py`](helper_scripts_for_data_preparation/clean_jsonl.py).
- **Non-ads**: impresso has no "non-ad" label. Articles were sampled from topic-model topics that are typically non-ad (topics were tagged by GPT and DeepSeek, see `cleaning_topics/` in git history before `b8c837c`). The small tuning and test sets were annotated by hand.
- **Cleaning via 4-fold CV** ([`cross-validation-cleaning/`](cross-validation-cleaning/)): the 3800 + 3800 set was split into 4 folds. A model was trained on 3 folds and predicted the 4th. Disagreements were inspected via article URLs ([`split_jsonl_agreement.py`](cross-validation-cleaning/split_jsonl_agreement.py), [`jsonl2url.py`](cross-validation-cleaning/jsonl2url.py)) and mislabelled items removed, leaving 3725 + 3725. That cleaned set trained v1.0.

### v2 data (Mar–May 2026)

Goal: more data and broader coverage across newspapers and years, with labels checked by an LLM and a human. Scripts and every intermediate file are in [`data_preparation_second_finetuning/`](data_preparation_second_finetuning/):

| Step | Script | Output | Rows |
|---|---|---|---|
| Sample v1 model predictions from S3 (`content-item-classification-base-multilingual_v1-0-0`), 1 ad + 1 non-ad per newspaper-year | `sample_balanced_articles.py` (*not in repo*) | `ads-nonads_inspection_sample.jsonl` (+ `_log.txt`) | 11784 |
| Attach text, drop empty or very short items | `filter_ft_jsonl.py` | `..._ft.jsonl` | 11184 |
| Ask an OpenAI model (`gpt-5.4-mini`) for ad / non-ad; split by agreement with v1 | `openai_compare_ad_classification.py` | `..._ft_agree_all.jsonl` / `..._ft_disagree_all.jsonl` | 6715 / 4471 |
| Filter the disagreements | `filter_ft_jsonl.py` | `..._disagree_filtered.jsonl` | 1732 |
| Newspaper-balanced samples for human review (300, then 550 more from the rest) | `sample_balanced_by_newspaper.py`, `remove_reviewed_from_filtered.py` | `..._sampled_300.jsonl`, `..._remaining_sampled_550.jsonl` | 850 |
| Export review sheet with article links, then convert back (rows marked "Unsure" dropped) | `export_sampled_jsonl_to_csv.py`, `clean_review_csv_to_jsonl.py` | `human_annotations.jsonl` (310 ad / 222 non-ad) | 532 |
| Combine agreed and human-reviewed rows | | `finetuning-2-ads-nonads.jsonl` (git history, `db1e738`) | 7247 |
| Merge with v1 data, balance classes | | `fine-tuning-data/{ads,non_ads}_8623_finetuning.jsonl` | 8623 + 8623 |

Composition of the final set: **non-ads** = 3720 from v1 + 4903 from the relabelling. **Ads** = 3709 from v1 + 2184 from the relabelling + **2730 from a source not recorded in the repo** (see loose ends).

The tuning set grew from 100 + 100 to 300 + 300; the original 100 are included. The test set is identical to v1's.

---

## Known issues and loose ends

- **Hub `pipeline.py` defaults are a mix of v1 and v2.** The published v2 `pipeline.py` uses v2's thresholds (`ad_threshold 0.9612`, `fr:0.8124, other:0.9570`). But it kept v1's `pool=logits_max`, `temperature=0.8`, `chunk_words=0`, `short_len=30`, while v2's tuning chose `logits_mean`, `1.2`, `200`, `50` ([`best_params.json`](best_params.json)). Thresholds tuned under one pooling and temperature setup aren't calibrated for another, so check this on the test set and republish if needed. The repo's `pipeline.py` mirrors the Hub as-is.
- **No v2 test-set evaluation is recorded.** Run step 3 above and add the numbers to the version table.
- **Origin of 2730 v2 ads is undocumented.** They aren't in the v1 set, the relabelling output, or any earlier file in git history. Most likely a fresh sample of impresso ads added to balance the classes.
- `sample_balanced_articles.py`, which made the initial v2 sample, isn't in the repo (its log is).
- `openai_compare_ad_classification.py` looks for `cookbook/lib` *inside* `data_preparation_second_finetuning/`. It silently falls back to no S3 transport params, which is fine for local files.
- `model_pushing.py` also uploads `best_params_final.json` *if it exists* (the pipeline would auto-load it). It is currently not shipped, so the defaults inside `pipeline.py` apply.
- `helper_scripts_for_data_preparation/non_ads_3800.jsonl` is a v1 working file kept for reference.

## Git history notes

Work happened on three branches, consolidated into `main` in Nov 2026:

- `main`: v1 code, cross-validation, HF packaging.
- `saving_work` (Oct 2025): snapshot of v1 working files (topic tagging, early data versions, calibration script).
- `clean-branch` (Mar–May 2026, built on `saving_work`): repo cleanup plus all v2 data work.

The old branches are kept as tags `archive/saving_work` and `archive/clean-branch`. Files removed during cleanup (e.g. `calibrate_finetuned_model.py`, `cleaning_topics/`, `non-ads-stuff/`, `ads_files/`) are recoverable from those tags.
