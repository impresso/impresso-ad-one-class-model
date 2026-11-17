# How the model is fine-tuned

How [`fine_tune_model.py`](../fine_tune_model.py) turns the genre classifier into an ad detector. What happens to the trained model at inference is covered in [how-ad-classification-works.md](how-ad-classification-works.md).

## Idea: one-vs-rest on the existing genre head

The base model [`classla/xlm-roberta-base-multilingual-text-genre-classifier`](https://huggingface.co/classla/xlm-roberta-base-multilingual-text-genre-classifier) (XLM-RoBERTa base, multilingual) already predicts 9 genres, one of which is **Promotion**. That class roughly means "ad", but it wasn't trained on historical newspaper ads.

Instead of replacing the head with a binary one, training keeps the 9-way head and teaches it **Promotion vs. everything else**:

- **every ad** gets label `Promotion` (id 8)
- **every non-ad** gets a *randomly drawn* non-Promotion label, weighted towards the most likely genres for non-ad newspaper text:

  | Genre | Weight | Share of non-ads |
  |---|---|---|
  | News, Information/Explanation | 3 | 21% each |
  | Opinion/Argumentation, Legal | 2 | 14% each |
  | Other, Instruction, Forum, Prose/Lyrical | 1 | 7% each |

These non-ad labels are synthetic; nobody checked which genre each text really is. So after training, only **P(Promotion)** is meaningful. The other 8 outputs are no longer reliable genre predictions, and the inference code uses nothing but P(Promotion).

Why do it this way: the model ships with no architecture change, so the same code loads it as the base model, and the pretrained genre knowledge is a better starting point than a freshly initialised binary head.

## Input data

Two JSONL files, one with ads and one with non-ads. Only `ft` (text) is used. The class comes from which file a row is in. The current data is described in the [README](../README.md#where-the-data-came-from):

| | ads | non-ads |
|---|---|---|
| v2 training (`fine-tuning-data/v2/*_8623_finetuning.jsonl`) | 8623 | 8623 |

Steps:

1. Build examples as above. Rows with empty text are skipped.
2. **Stratified 85 / 15 split** into train and validation, by ad / non-ad (seed 42). For v2: 14,659 train / 2,587 validation.
3. Tokenise with padding and truncation to **512 tokens**. Longer texts are cut; there is no chunking during training.

The separate tuning set (300 + 300) and test set (100 + 100) are **not** used here.

## Training configuration

| Setting | Script default | **v2 run** |
|---|---|---|
| epochs (max) | 3 | **5** (early-stopped at 2.0; best checkpoint at 1.3) |
| learning rate | 2e-5 | **1.5e-5** |
| batch size (eval = 2×) | 16 | 16 |
| warmup ratio | 0.1 | 0.1 |
| weight decay | 0.01 | 0.01 |
| eval / save every | 200 steps | 200 steps |
| early stopping | patience 3 evals on F1 | same |
| best checkpoint | highest validation **binary F1** (reloaded at the end) | same |
| checkpoints kept | 3 | 3 |
| seed | 42 | 42 |

Device: MPS on Apple Silicon if available, otherwise CUDA (with fp16), otherwise CPU.

## Validation metrics

Each evaluation computes:

- **Binary metrics (the ones that matter):** prediction = argmax is Promotion vs. label is Promotion. Reported as precision, recall, F1, binary accuracy and TP/FP/TN/FN. F1 selects the best checkpoint and drives early stopping.
- **Multi-class accuracy:** exact 9-way label match. Since non-ad labels are random, this number is meaningless and can be ignored.

At the end the script prints a binary classification report and confusion matrix on the validation split.

Validation uses argmax without any thresholds or rules, so it measures the bare model. Real performance comes from the tuned decision layer, measured with `evaluate_model.py` on the test set.

## Output

`--output_dir` (e.g. `fine_tuned_xlm_2/`) contains:

| File | |
|---|---|
| `model.safetensors`, `config.json` | best checkpoint's weights and config (9-label head) |
| tokenizer files | copied from the base model |
| `training_info.json` | base model, Promotion id, label map, train/val sizes, epochs, LR, batch size |
| `checkpoint-*/` | last 3 checkpoints (not needed for inference; excluded when publishing) |

## Next steps

1. Tune the decision layer: `optimize_hyperparams.py --model <output_dir>` → `best_params.json`.
2. Evaluate on the test set: `evaluate_model.py`.
3. Publish: `model_pushing.py` (uploads weights and tokenizer, plus `pipeline.py`).

Commands are in the [README](../README.md#workflow-reproducing-v2).
