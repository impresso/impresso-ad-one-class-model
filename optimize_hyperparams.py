#!/usr/bin/env python3
"""
Hyperparameter optimization that uses model_approach.py script directly.
Uses internal caching for speed during search, validates with model_approach.py.

Usage:
python optimize_hyperparams.py \
  --ads ads_100_for_hyperparameters.jsonl \
  --non_ads non_ads_100_for_hyperparameters.jsonl \
  --output best_params.json \
  --max_configs 120

This version:
- Caches model logits internally for fast configuration search
- Uses model_approach.py only for final validation of top candidates
- Ensures changes to model_approach.py are picked up in final results
"""

import argparse, json, time, random, tempfile, subprocess, os, re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedShuffleSplit

# ----------------------------
# Utility: IO and Text Processing
# ----------------------------
def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Text normalization (aligned to model_approach.py)
NORM_SPACES = re.compile(r"\s{2,}")
def normalize_text(t: str) -> str:
    if not isinstance(t, str):
        t = str(t)
    t = re.sub(r"[_`~^]+", " ", t)
    t = re.sub(r"\s+([.,:;!?()])", r"\1", t)
    t = NORM_SPACES.sub(" ", t)
    return t.strip()

def chunk_words(text: str, max_words: int):
    if max_words <= 0:
        return [text]
    ws = text.split()
    parts = [" ".join(ws[i:i+max_words]) for i in range(0, len(ws), max_words)]
    return parts if parts else [""]

# Simple heuristics (aligned with model_approach.py)
PHONE = re.compile(r"(?:\+?\d{2,3}[\s./-]?)?(?:\(?0\d{1,3}\)?[\s./-]?)\d(?:[\d\s./-]{5,})")
PRICE = re.compile(
    r"(?:CHF|SFr\.?|Fr\.?|fr\.?|€|\$)\s?\d{1,3}(?:[''`\s]?\d{3})*(?:\.-|[.-])?"
    r"|(?:\d{1,3}(?:[''`\s]?\d{3})*)(?:\s?(?:CHF|SFr\.?|Fr\.?|€|\$))(?:\.-|[.-])?",
    re.I
)
CUES_FR = r"(?:à\s?vendre|a\s?vendre|à\s?louer|a\s?louer|à\s?remettre|prix\s+à\s+discuter|écrire\s+à|sous\s+chiffres|tél\.?|loyer|charges|villa|attique|expertisée)"
CUES_DE = r"(?:zu\s?verkaufen|zu\s?vermieten|Preis|Schreib(?:en)?\s+an|unter\s+Chiffre|Tel\.?|Miete|Zimmer|Attika|expertisiert)"
CUES_LB = r"(?:ze\s?verkafen|ze\s?verlounen|Präis|Annonce|Tel\.?)"
CUES = re.compile(fr"\b(?:{CUES_FR}|{CUES_DE}|{CUES_LB})\b", re.I)

def rule_flags(t: str):
    return {
        "has_phone": bool(PHONE.search(t)),
        "has_price": bool(PRICE.search(t)),
        "has_cue": bool(CUES.search(t)),
        "len_words": len(t.split()),
        "pct_digits": (sum(ch.isdigit() for ch in t) / max(len(t),1)),
    }

def lang_len_threshold(lang: str, n_words: int, per_lang_thr: Dict[str,float], default_thr: float, short_bonus: float, short_len: int):
    base = per_lang_thr.get((lang or "").lower(), default_thr)
    if n_words <= short_len:
        return max(0.0, base - short_bonus)
    return base

# ----------------------------
# Data loading
# ----------------------------
def load_ground_truth(ads_file, non_ads_file=None):
    """Load and combine ground truth data from mixed file or separate files."""
    data = []
    if non_ads_file is None or ads_file == non_ads_file:
        with open(ads_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    item_type = (item.get('type') or "").lower()
                    if item_type in ['ad', 'advertisement', 'promotion']:
                        item['ground_truth'] = True
                    elif item_type in ['non-ad', 'article', 'news', 'non_ad', 'nonad']:
                        item['ground_truth'] = False
                    else:
                        continue
                    data.append(item)
    else:
        with open(ads_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    item['ground_truth'] = True
                    data.append(item)
        with open(non_ads_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    item['ground_truth'] = False
                    data.append(item)
    return data

# ----------------------------
# Fast model cache for hyperparameter search
# ----------------------------
class LogitCache:
    """
    For each (chunk_words, max_length), stores:
      - per-document chunk logits (np.array [n_chunks, n_labels])
      - per-document chunk word lengths (list[int])
    """
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self._tok = None
        self._model = None
        self.id2label = None
        self.promo_id = None
        # cache keyed strictly by (chunk_words, max_length)
        self.cache = {}  # key: (chunk_words, max_length) -> dict(doc_index -> (logits, lens))

    def _ensure_model(self):
        if self._tok is None or self._model is None:
            print("Loading model for caching...")
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device).eval()
            self.id2label = self._model.config.id2label
            self.promo_id = None
            for i, lab in self.id2label.items():
                if lab.lower() == "promotion":
                    self.promo_id = i
                    break
            if self.promo_id is None:
                raise RuntimeError("Could not find 'Promotion' label in model.config.id2label")

    def build_for_combo(self, docs: List[dict], chunk_words_val: int, max_length: int, batch_size: int):
        key = (chunk_words_val, max_length)
        if key in self.cache:
            return

        self._ensure_model()
        per_doc_logits = {}
        per_doc_lens = {}

        # Prepare all chunks
        all_texts = []
        mapping = []  # (doc_idx, n_chunks, lens_list)
        for i, d in enumerate(docs):
            txt = normalize_text(d.get("ft",""))
            parts = chunk_words(txt, chunk_words_val)
            lens = [len(p.split()) for p in parts]
            if not parts:
                parts = [""]
                lens = [0]
            mapping.append((i, len(parts), lens))
            all_texts.extend(parts)

        # Batched encoding
        tok = self._tok
        model = self._model
        device = self.device

        def batched(iterable, n):
            for idx in range(0, len(iterable), n):
                yield idx, iterable[idx:idx+n]

        logits_list = []
        with torch.no_grad():
            for _, batch in batched(all_texts, batch_size):
                enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                enc = {k:v.to(device) for k,v in enc.items()}
                logits = model(**enc).logits  # (B, n_labels)
                logits_list.append(logits.cpu().numpy())

        all_logits = np.vstack(logits_list) if logits_list else np.zeros((0, len(self.id2label)))
        # Slice back to documents
        pos = 0
        for doc_idx, n_chunks, lens in mapping:
            L = all_logits[pos:pos+n_chunks]
            per_doc_logits[doc_idx] = L
            per_doc_lens[doc_idx] = lens
            pos += n_chunks

        self.cache[key] = {"logits": per_doc_logits, "lens": per_doc_lens}

    def get_doc(self, combo, doc_idx):
        if combo not in self.cache:
            raise KeyError(f"Cache miss for combo={combo}. Cached keys: {list(self.cache.keys())}")
        pack = self.cache[combo]
        return pack["logits"][doc_idx], pack["lens"][doc_idx]

# ----------------------------
# Pooling & scoring on cached logits
# ----------------------------
def pool_probs_from_logits(L: np.ndarray, lens: List[int], pool: str, temperature: float) -> np.ndarray:
    if L.ndim == 1:
        L = L[None, :]
    if pool == "max":
        pooled_probs = torch.softmax(torch.tensor(L), dim=-1).max(dim=0).values.numpy()
    elif pool == "mean":
        pooled_probs = torch.softmax(torch.tensor(L), dim=-1).mean(dim=0).numpy()
    elif pool == "logits_max":
        pooled_logits = L.max(axis=0)
        pooled_logits = pooled_logits / max(temperature, 1e-6)
        pooled_probs = torch.softmax(torch.tensor(pooled_logits), dim=-1).numpy()
    elif pool == "logits_mean":
        pooled_logits = L.mean(axis=0)
        pooled_logits = pooled_logits / max(temperature, 1e-6)
        pooled_probs = torch.softmax(torch.tensor(pooled_logits), dim=-1).numpy()
    else:  # logits_weighted
        w = np.array(lens, dtype=np.float32)
        w = w / (w.sum() + 1e-9)
        pooled_logits = (L * w[:, None]).sum(axis=0)
        pooled_logits = pooled_logits / max(temperature, 1e-6)
        pooled_probs = torch.softmax(torch.tensor(pooled_logits), dim=-1).numpy()
    return pooled_probs

# ----------------------------
# Model runner using model_approach.py (for final validation)
# ----------------------------
def run_model_approach(input_file: str, params: dict, timeout: int = 300) -> List[dict]:
    """Run model_approach.py with given parameters and return results."""
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as tmp_out:
        output_file = tmp_out.name
    
    # Build command
    cmd = ['python', 'model_approach.py', '--in', input_file, '--out', output_file]
    
    # Add parameters
    for key, value in params.items():
        if key == 'model':
            cmd.extend(['--model', str(value)])
        elif key == 'batch_size':
            cmd.extend(['--batch_size', str(value)])
        elif key == 'max_length':
            cmd.extend(['--max_length', str(value)])
        elif key == 'chunk_words':
            cmd.extend(['--chunk_words', str(value)])
        elif key == 'pool':
            cmd.extend(['--pool', str(value)])
        elif key == 'ad_threshold':
            cmd.extend(['--ad_threshold', str(value)])
        elif key == 'lang_thresholds' and value:
            cmd.extend(['--lang_thresholds', str(value)])
        elif key == 'short_len':
            cmd.extend(['--short_len', str(value)])
        elif key == 'short_bonus':
            cmd.extend(['--short_bonus', str(value)])
        elif key == 'min_words':
            cmd.extend(['--min_words', str(value)])
        elif key == 'temperature':
            cmd.extend(['--temperature', str(value)])
        elif key == 'meta_clf' and value:
            cmd.extend(['--meta_clf', str(value)])
    
    try:
        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode != 0:
            print(f"Error running model_approach.py: {result.stderr}")
            return []
        
        # Read results
        results = []
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"model_approach.py timed out after {timeout}s")
        return []
    except Exception as e:
        print(f"Error running model_approach.py: {e}")
        return []
    finally:
        # Clean up
        if os.path.exists(output_file):
            os.unlink(output_file)

# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray):
    # IMPORTANT: zero_division=0 prevents UndefinedMetricWarning when a class has no predicted samples
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0,1], zero_division=0
    )  # 0=non-ad, 1=ad
    macro_f1 = f1.mean()
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    return {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
        "precision_non_ad": float(p[0]),
        "recall_non_ad": float(r[0]),
        "f1_non_ad": float(f1[0]),
        "precision_ad": float(p[1]),
        "recall_ad": float(r[1]),
        "f1_ad": float(f1[1]),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
        "class_support": {"non_ad": int(support[0]), "ad": int(support[1])},
    }

# ----------------------------
# Threshold sweep
# ----------------------------
def best_threshold_for_metric(y_true, scores, metric="macro_f1"):
    """Find best single threshold for the given metric."""
    uniq = np.unique(scores)
    if len(uniq) > 400:
        qs = np.linspace(0.02, 0.98, 200)
        thr_list = np.quantile(scores, qs)
    else:
        thr_list = uniq

    best_val = -1.0
    best_thr = 0.5
    best_metrics = None
    for thr in thr_list:
        y_pred = (scores >= thr)
        m = compute_metrics(y_true, y_pred, scores)
        val = m["macro_f1"] if metric=="macro_f1" else m["balanced_accuracy"]
        if val > best_val:
            best_val = val
            best_thr = float(thr)
            best_metrics = m
    return best_thr, best_metrics

def best_thresholds_per_language(y_true, scores, langs, metric="macro_f1", min_lang=50):
    """Find best per-language thresholds."""
    global_thr, _ = best_threshold_for_metric(y_true, scores, metric)
    lang_thr = {}
    y_pred = np.zeros_like(y_true, dtype=bool)

    for lg in set(langs):
        idx = np.array([i for i, L in enumerate(langs) if L == lg])
        if len(idx) >= min_lang:
            thr, _ = best_threshold_for_metric(y_true[idx], scores[idx], metric)
            lang_thr[lg] = float(thr)

    for i in range(len(y_true)):
        lg = langs[i]
        thr = lang_thr.get(lg, global_thr)
        y_pred[i] = scores[i] >= thr

    metrics = compute_metrics(y_true, y_pred, scores)
    return lang_thr, float(global_thr), metrics

# ----------------------------
# Config search
# ----------------------------
def define_search_space():
    return {
        "pool": ["logits_weighted", "logits_mean", "logits_max"],
        "temperature": [0.8, 1.0, 1.2, 1.5],
        "short_bonus": [0.15, 0.20, 0.25],
        "short_len": [30, 40, 50],
        "min_words": [0, 3, 5],
        "chunk_words": [0, 200],
        "max_length": [512],
        "threshold_mode": ["global", "per_lang"],
    }

def sample_configs(space: Dict[str, List], max_configs: int, seed: int = 42):
    random.seed(seed)
    keys = list(space.keys())
    from itertools import product
    all_cfgs = []
    for vals in product(*[space[k] for k in keys]):
        cfg = {k: v for k, v in zip(keys, vals)}
        all_cfgs.append(cfg)
    random.shuffle(all_cfgs)
    return all_cfgs[:max_configs]

# ----------------------------
# Fast evaluation using cached logits
# ----------------------------
def eval_config_fast(cfg, docs, ground_truth, langs, cache, objective_metric):
    """Fast evaluation using cached logits."""
    cw, ml = cfg["chunk_words"], cfg["max_length"]
    key = (cw, ml)
    pool = cfg["pool"]
    T = float(cfg["temperature"])
    short_bonus = float(cfg["short_bonus"])
    short_len = int(cfg["short_len"])
    min_words = int(cfg["min_words"])
    thr_mode = cfg["threshold_mode"]

    promo_scores = []
    used_ground_truth = []
    used_langs = []
    pre_flags = [rule_flags(normalize_text(d.get("ft",""))) for d in docs]

    for i, (doc, gt, lg) in enumerate(zip(docs, ground_truth, langs)):
        if min_words and pre_flags[i]["len_words"] < min_words:
            continue
        try:
            L, lens = cache.get_doc(key, i)
        except KeyError:
            # This shouldn't happen if we cached properly, but handle gracefully
            return None
        probs = pool_probs_from_logits(L, lens, pool, T)
        promo_prob = float(probs[cache.promo_id])
        promo_scores.append(promo_prob)
        used_ground_truth.append(gt)
        used_langs.append(lg)

    if not promo_scores:
        return None

    promo_scores = np.array(promo_scores)
    used_ground_truth = np.array(used_ground_truth, dtype=bool)
    used_langs = np.array(used_langs)

    # Optimize thresholds
    if thr_mode == "per_lang":
        per_lang_thr, global_thr, metrics = best_thresholds_per_language(
            used_ground_truth, promo_scores, used_langs, metric=objective_metric, min_lang=30
        )
        lang_thr_str = ",".join(f"{lg}:{thr:.4f}" for lg, thr in per_lang_thr.items())
    else:
        per_lang_thr = {}
        global_thr, metrics = best_threshold_for_metric(used_ground_truth, promo_scores, objective_metric)
        lang_thr_str = ""

    return {
        "metrics": metrics,
        "global_threshold": float(global_thr),
        "lang_thresholds": lang_thr_str,
        "effective_samples": int(len(used_ground_truth))
    }

# ----------------------------
# Validation using model_approach.py
# ----------------------------
def validate_config_with_model_approach(cfg, docs, ground_truth, langs, model_name, timeout, objective_metric):
    """Validate a configuration by calling model_approach.py."""
    
    # Create temporary input file (without ground_truth for model)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as tmp_in:
        input_file = tmp_in.name
        for doc in docs:
            # Remove ground_truth field for model input
            model_input = {k: v for k, v in doc.items() if k != 'ground_truth'}
            tmp_in.write(json.dumps(model_input, ensure_ascii=False) + '\n')
    
    try:
        # Prepare parameters (use optimized thresholds from fast evaluation)
        params = {
            "model": model_name,
            "batch_size": 16,
            "max_length": cfg["max_length"],
            "chunk_words": cfg["chunk_words"],
            "pool": cfg["pool"],
            "temperature": cfg["temperature"],
            "short_len": cfg["short_len"],
            "short_bonus": cfg["short_bonus"],
            "min_words": cfg["min_words"],
            "ad_threshold": cfg["ad_threshold"],  # Use optimized threshold
        }
        
        if cfg.get("lang_thresholds"):
            params["lang_thresholds"] = cfg["lang_thresholds"]
        
        # Run model
        results = run_model_approach(input_file, params, timeout)
        
        if not results:
            return None
        
        # Extract predictions and compare with ground truth
        y_pred = []
        y_true = []
        
        for i, result in enumerate(results):
            if i >= len(ground_truth):
                break
                
            # Skip if model marked as skipped
            if result.get('note') == 'skipped_min_words':
                continue
                
            pred = result.get('is_ad_pred', False)
            y_pred.append(pred)
            y_true.append(ground_truth[i])
        
        if not y_pred:
            return None
            
        y_pred = np.array(y_pred, dtype=bool)
        y_true = np.array(y_true, dtype=bool)
        
        # Compute metrics for this configuration
        metrics = compute_metrics(y_true, y_pred, np.array([0.5] * len(y_pred)))  # dummy scores
        
        return {
            "metrics": metrics,
            "effective_samples": int(len(y_true))
        }
        
    except Exception as e:
        print(f"Error validating config: {e}")
        return None
    finally:
        # Clean up
        if os.path.exists(input_file):
            os.unlink(input_file)

# ----------------------------
# End-to-end optimization
# ----------------------------
def optimize(args):
    # Load data
    data = load_ground_truth(args.ads, args.non_ads)
    if not data:
        raise SystemExit("No data loaded. Check your input files.")

    docs = []
    ground_truth = []
    langs = []
    
    for i, item in enumerate(data):
        row = dict(item)
        row["id"] = item.get("id", i)
        lg = (item.get("lg") or item.get("lang") or "").lower()
        langs.append(lg if lg else "other")
        ground_truth.append(bool(item["ground_truth"]))
        docs.append(row)
    
    ground_truth = np.array(ground_truth, dtype=bool)
    langs = np.array(langs)
    
    n_ads = int(ground_truth.sum())
    n_non = int((~ground_truth).sum())
    print(f"Loaded {len(docs)} samples | Ads={n_ads} Non-ads={n_non}")

    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate configurations
    space = define_search_space()
    configs = sample_configs(space, args.max_configs, seed=args.seed)
    print(f"Testing up to {len(configs)} configurations")

    # Create cache and pre-compute logits for unique model combos
    unique_model_passes = sorted(set((c["chunk_words"], c["max_length"]) for c in configs))
    cache = LogitCache(args.model, device)

    print(f"Pre-caching logits for {len(unique_model_passes)} model configurations...")
    for i, (cw, ml) in enumerate(unique_model_passes):
        print(f"  [{i+1}/{len(unique_model_passes)}] Caching combo: chunk_words={cw}, max_length={ml}")
        t0 = time.time()
        cache.build_for_combo(docs, cw, ml, batch_size=args.cache_batch_size)
        print(f"    Cached in {time.time()-t0:.1f}s")

    # Create stratified subset for coarse evaluation
    if len(docs) > args.subsample_for_coarse:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.subsample_for_coarse, random_state=args.seed)
        for _, idx in sss.split(np.zeros(len(docs)), ground_truth.astype(int)):
            subset_idx = sorted(idx)
            break
        subset_docs = [docs[i] for i in subset_idx]
        subset_ground_truth = ground_truth[subset_idx]
        subset_langs = langs[subset_idx]
    else:
        subset_idx = list(range(len(docs)))
        subset_docs = docs
        subset_ground_truth = ground_truth
        subset_langs = langs

    print(f"\n== Fast coarse evaluation on {len(subset_docs)} samples (stratified subset)")
    
    # Fast coarse evaluation using cache
    coarse_results = []
    for i, cfg in enumerate(configs, 1):
        if i % 10 == 0:
            print(f"  Testing config {i}/{len(configs)}...")
        
        res = eval_config_fast(cfg, subset_docs, subset_ground_truth, subset_langs, cache, args.objective_metric)
        
        if res is None:
            continue
            
        score = res["metrics"]["macro_f1"] if args.objective_metric=="macro_f1" else res["metrics"]["balanced_accuracy"]
        coarse_results.append((score, cfg, res))

    if not coarse_results:
        raise SystemExit("No valid configurations evaluated in coarse phase.")

    # Select top configurations and add optimized thresholds
    coarse_results.sort(key=lambda x: x[0], reverse=True)
    keep = min(args.top_k_full_eval, len(coarse_results))
    
    finalists = []
    for i, (score, cfg, res) in enumerate(coarse_results[:keep]):
        # Add optimized thresholds to config for final validation
        enhanced_cfg = dict(cfg)
        enhanced_cfg["ad_threshold"] = res["global_threshold"]
        enhanced_cfg["lang_thresholds"] = res["lang_thresholds"]
        finalists.append(enhanced_cfg)
        print(f"  {i+1}. Score: {score:.4f} - {cfg}")

    print(f"\nSelected top {keep} configs for validation with model_approach.py")

    # Final validation using model_approach.py
    print(f"\n== Final validation on all {len(docs)} samples using model_approach.py")
    best_score = -1.0
    best_pack = None
    all_logs = []

    for rank, cfg in enumerate(finalists, 1):
        print(f"\n[{rank}/{len(finalists)}] Validating with model_approach.py...")
        t0 = time.time()
        
        res = validate_config_with_model_approach(
            cfg, docs, ground_truth, langs, 
            args.model, args.timeout, args.objective_metric
        )
        
        elapsed = time.time() - t0
        
        if res is None:
            print(f"  -> Failed")
            continue
            
        score = res["metrics"]["macro_f1"] if args.objective_metric=="macro_f1" else res["metrics"]["balanced_accuracy"]
        
        log = {
            "rank_in_finalists": rank,
            "params": cfg,
            "result": res,
            "elapsed_sec": elapsed
        }
        all_logs.append(log)
        
        print(f"  -> Score: {score:.4f}")
        print(f"     F1_non_ad: {res['metrics']['f1_non_ad']:.4f}")
        print(f"     F1_ad: {res['metrics']['f1_ad']:.4f}")
        print(f"     Balanced_acc: {res['metrics']['balanced_accuracy']:.4f}")
        print(f"     MCC: {res['metrics']['mcc']:.4f}")
        print(f"     Time: {elapsed:.1f}s")

        if score > best_score:
            best_score = score
            best_pack = (cfg, res)

    if best_pack is None:
        raise SystemExit("No successful configurations found!")

    best_cfg, best_res = best_pack

    # Prepare output compatible with model_approach.py
    output_params = dict(best_cfg)
    output_params.pop("threshold_mode", None)  # model_approach doesn't know this flag
    output_params.setdefault("model", args.model)

    # Save results
    out_blob = {
        "best_params": output_params,
        "best_metrics": best_res["metrics"],
        "notes": {
            "objective_metric": args.objective_metric,
            "dataset_info": {
                "total_samples": len(docs),
                "ads_count": int(ground_truth.sum()),
                "non_ads_count": int((~ground_truth).sum())
            },
            "optimization_method": "cached_search_with_model_approach_validation"
        }
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_blob, f, indent=2, ensure_ascii=False)
    print(f"\nSaved best parameters to: {args.output}")

    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed optimization log to: {args.log}")

    # Generate command line
    parts = ["python model_approach.py --in INPUT.jsonl --out OUTPUT.jsonl"]
    for k, v in output_params.items():
        if k in {"model"}:
            parts.append(f"--{k} {v}")
        elif k == "lang_thresholds" and v:
            parts.append(f'--lang_thresholds "{v}"')
        else:
            parts.append(f"--{k} {v}")
    
    print("\nOptimal command line:")
    print(" ".join(parts))

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Hybrid hyperparameter optimization: fast caching + model_approach.py validation")
    ap.add_argument('--ads', required=True, help='JSONL file with ads (or mixed file with type labels)')
    ap.add_argument('--non_ads', help='JSONL file with non-ads (optional if using mixed file)')
    ap.add_argument('--output', default='best_params.json', help='Output file for best parameters')
    ap.add_argument('--log', default='optimization_log.json', help='Detailed log of finalists')

    # Model and runtime
    # ap.add_argument('--model', default='classla/xlm-roberta-base-multilingual-text-genre-classifier', help='HF model id')
    ap.add_argument('--model', default='fine_tuned_xlm', help='HF model id')
    ap.add_argument('--max_configs', type=int, default=100, help='Max configs to explore')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cache_batch_size', type=int, default=16, help='Batch size for caching')
    ap.add_argument('--timeout', type=int, default=300, help='Timeout for model_approach.py validation calls')

    # Successive halving controls
    ap.add_argument('--subsample_for_coarse', type=int, default=300, help='Subset size for coarse evaluation (stratified)')
    ap.add_argument('--top_k_full_eval', type=int, default=10, help='How many top configs to validate with model_approach.py')

    # Objective metric
    ap.add_argument('--objective_metric', choices=['macro_f1','balanced_accuracy'], default='macro_f1',
                    help='Primary score to optimize (imbalance-aware)')

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    optimize(args)
