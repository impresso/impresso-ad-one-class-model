# How ad classification works (inference)

How a text becomes **AD** or **NOT_AD** once a fine-tuned model exists. Training is covered in [how-the-model-is-fine-tuned.md](how-the-model-is-fine-tuned.md).

The same logic is implemented twice. Keep the two in sync when changing anything:

| File | Used by |
|---|---|
| [`model_approach.py`](../model_approach.py) | CLI (JSONL in → JSONL out). Called by `optimize_hyperparams.py` and `evaluate_model.py`. |
| [`pipeline.py`](../pipeline.py) | Hugging Face custom pipeline shipped with the model on the Hub (`trust_remote_code=True`). |

## Pipeline at a glance

```
text ──► normalise ──► chunk (optional) ──► fine-tuned XLM-R ──► pool chunks + temperature
                                                                     │
                                                          P(Promotion) = promo_prob
                                                                     │
 threshold: per-language ─► short-text bonus ─► adaptive factors     │
                                │                                    ▼
                                └──────────► rule adjustments (prices, phones, cues, non-ad signals)
                                                                     │
                                                     AD  if  final_prob ≥ threshold
```

The model provides the main signal. The rules mostly affect texts where the model is unsure.

## 1. Parameters

These are set per run (CLI flags or pipeline kwargs) and chosen by `optimize_hyperparams.py`.

| Parameter | Meaning | v2 tuned value ([`best_params.json`](../best_params.json)) |
|---|---|---|
| `chunk_words` | split text into chunks of N words; `0` = no chunking (the tokenizer truncates at `max_length`) | 200 |
| `max_length` | max tokens per chunk | 512 |
| `pool` | how chunk outputs are combined (see §2) | `logits_mean` |
| `temperature` | logits are divided by T before softmax; higher T means softer probabilities | 1.2 |
| `ad_threshold` | fallback threshold for any language not in `lang_thresholds` | 0.9612 |
| `lang_thresholds` | per-language thresholds, `lang:thr,...` | `fr:0.8124,other:0.9570` |
| `short_len`, `short_bonus` | texts with ≤ `short_len` words get the threshold lowered by `short_bonus` | 50, 0.2 |
| `min_words` | skip texts with fewer words (`label: SKIPPED`, `is_ad_pred: null`) | 0 |
| `meta_clf` | optional scikit-learn stacker (none has been trained; leave empty) | — |

> ⚠️ The `pipeline.py` currently on the Hub has v2 thresholds but v1 settings for `pool` / `temperature` / `chunk_words` / `short_len` (`logits_max`, 0.8, 0, 30). Pass the v2 values explicitly, or fix the defaults in `pipeline.py` and republish.

Language comes from the row's `lg` field. The tuned map only has `fr` and `other`, because tuning needs at least 30 examples per language. Rows with `lg: null` or `de`/`lb` don't match the `other` key literally, so they fall back to `ad_threshold`. That value (0.9612) is almost the same as the `other` threshold.

## 2. Model scoring

1. **Normalise**: collapse whitespace, drop OCR junk characters (`_ ` ~ ^`), remove spaces before punctuation.
2. **Chunk** into `chunk_words`-word pieces, each truncated to `max_length` tokens.
3. **Pool** the per-chunk outputs into one 9-class distribution:

   | `pool` | Computation |
   |---|---|
   | `logits_mean` | mean of chunk logits → /T → softmax |
   | `logits_max` | element-wise max of chunk logits → /T → softmax |
   | `logits_weighted` | chunk logits weighted by chunk word count → /T → softmax |
   | `mean` / `max` | softmax per chunk, then mean / max (ignores T) |

4. `promo_prob` = probability of the `Promotion` class. Also recorded: `xgenre_top_label` and `xgenre_top_prob`. After fine-tuning, the non-Promotion genres are not real genre predictions (see the fine-tuning doc).

> **Ensemble signal = no-op.** The code blends `0.85·promo_prob + 0.15·ensemble_signal`, where the ensemble sums labels named like Commercial, Business, etc. The model has no such labels; its 9 labels are Other, Information/Explanation, News, Instruction, Opinion/Argumentation, Forum, Prose/Lyrical, Legal, Promotion. So `ensemble_signal == promo_prob`, and the blend equals `promo_prob`.

## 3. Threshold

The threshold starts from the language threshold, minus `short_bonus` for short texts. It is then multiplied by **adaptive factors** that make number-heavy, ad-shaped texts easier to flag:

| Signal | Factor |
|---|---|
| < 10 words / < 25 words / > 100 words | ×1.10 / ×1.05 / ×0.95 |
| digit share > 15% / > 8% | ×0.92 / ×0.96 |
| contact-like pattern (`: 12`, `contact:`, `12-34-56`) | ×0.94 |
| ≥ 2 price/currency mentions | ×(1 − 0.04·n), floored at 0.88 |

## 4. Rule features

Regexes run on the normalised text. Weighted hits give a `rule_score`; `rule_hit = rule_score ≥ 1.5`.

| Feature | Weight | Detects |
|---|---|---|
| `has_price` | 2.0 | `CHF / SFr. / Fr. / € / $ / USD / EUR` next to a number (`Fr. 2'100.–`, `450 CHF`) |
| `has_phone` | 2.0 | phone-number shapes (`079 123 45 67`, `+41 22 ...`) |
| `has_cue` | 1.5 | ad words. FR: *à vendre, à louer, sous chiffres, tél., loyer, occasion…* DE: *zu verkaufen, zu vermieten, Preis, unter Chiffre, Miete, Zimmer…* LB: *ze verkafen, ze verlounen, Präis, Annonce…* |
| `has_area` | 1.0 | `85 m²`, `120 m2` |
| `has_rooms` | 1.0 | `3 pièces` (FR only) |
| `has_address` | 0.8 | Rue, Avenue, Platz, Strasse, Place… |
| `has_zip` | 0.5 | any 4-digit number (Swiss ZIP; also matches years) |
| `has_year` | 0.3 | 19xx / 20xx |
| `has_km` | 0.3 | `120'000 km` |

`rule_confidence = min(1, 0.4·#strong + 0.2·#medium + 0.1·#weak)`. Strong = price, phone. Medium = cue, area, rooms. Weak = address, zip, year, km.

**Non-ad signals** reduce false positives. Keyword groups are checked: news/obituary (*communiqué, avis, décès, nécrologie…*), official (*commune, conseil, canton…*), event (*concert, exposition, assemblée…*), legal (*tribunal, notaire, succession…*), plus a top genre of `News` or `Legal`.

```
non_ad_confidence = min(1, 0.3·#keyword_groups_hit + 0.5·[top label is News/Legal])
precision_penalty = 0.2 · non_ad_confidence        # 0 … 0.2
```

## 5. How rules change the score

Rules count for more when the model is unsure:

```
model_confidence   = |promo_prob − 0.5| · 2         # 0 at p=0.5, 1 at p=0 or 1
rule_influence     = 0.3 + 1.2 · (1 − model_confidence)   # 1.5 at p=0.5 … 0.3 at p=0/1
```

| promo_prob | 0.5 | 0.6 / 0.4 | 0.7 / 0.3 | 0.8 / 0.2 | 0.9 / 0.1 |
|---|---|---|---|---|---|
| rule_influence | 1.50 | 1.26 | 1.02 | 0.78 | 0.54 |

Applied in this order (when no `meta_clf` is used):

1. **Rule boosts.** Only when 0.125 < promo_prob < 0.875.
   - Very strong rules (confidence > 0.7, score ≥ 4) or strong rules (> 0.5, ≥ 3) with a low penalty: push the score to just above the threshold. This flips the decision to AD.
   - Medium rules: if below the threshold, add `(0.65·gap + 0.08 − penalty) · influence`, which often reaches the threshold.
   - Weak rules: small boost.
2. **Combination boosts.** Same probability band. Price + phone, price + cue, phone + cue, price + address + ZIP, and cue + area + rooms each add a boost (largest first). The result is capped at 0.92.
3. **Threshold reduction.** When 0.175 < promo_prob < 0.825 and rule_score ≥ 3.5 (or ≥ 2.5 with a very low penalty): the threshold is multiplied by (1 − 0.18·influence) or (1 − 0.12·influence).
4. **Safety damping.** A model very sure it's *not* an ad (promo_prob < 0.075) has its score shrunk further. A strong non-ad penalty with promo_prob < 0.35 shrinks the score too.
5. **Promotion top label.** If `Promotion` is the top genre and there's a rule hit, the score is pushed just above the threshold. Without a rule hit it is multiplied by about 1.12–1.30.
6. **Decision:** `is_ad = final_prob ≥ threshold`.

Every constant in this section is hand-set. None of them is tuned by `optimize_hyperparams.py`.

## 6. Output

| Field (`pipeline.py` / `model_approach.py`) | Meaning |
|---|---|
| `label` / `is_ad_pred` | `AD`/`NOT_AD` (or `SKIPPED`) / `true`/`false` |
| `score` / `promotion_prob_final` | final score after rules |
| `promotion_prob` | raw pooled P(Promotion) |
| `threshold_used` | effective threshold after all adjustments |
| `xgenre_top_label`, `xgenre_top_prob` | top class of the 9-way head |
| diagnostics | rule flags, `rule_score`, `rule_confidence`, `model_confidence`, non-ad flags (`model_approach.py` also writes `base_threshold`, `precision_penalty`, `rule_influence_multiplier`) |

## 7. How the parameters are tuned

`optimize_hyperparams.py` searches this grid (sampled up to `--max_configs`):

| | values |
|---|---|
| `pool` | logits_weighted, logits_mean, logits_max |
| `temperature` | 0.8, 1.0, 1.2, 1.5 |
| `short_bonus` | 0.15, 0.20, 0.25 |
| `short_len` | 30, 40, 50 |
| `min_words` | 0, 3, 5 |
| `chunk_words` | 0, 200 |
| threshold mode | one global threshold, or per language (languages with ≥ 30 examples) |

1. **Fast search.** Model logits are cached once per `chunk_words` setting. For each config the best threshold(s) for **macro-F1** are found on the raw `promo_prob`. This stage ignores rules and the short-text bonus.
2. **Validation.** The top configs are re-run through `model_approach.py` with all rules. The best one is written to `best_params.json`, and all finalists go to `optimization_log.json`.

Because thresholds are picked on the raw probability, they come out very high (v2: 0.96 overall, 0.81 for FR). The rules then mainly rescue ads the model scored below that.
