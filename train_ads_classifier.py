# train_ads_classifier_json.py
# Two-stage Ads detector but reading JSON/JSONL instead of CSV.
# Stage A: char TF-IDF + Linear (optional but kept here as in original)
# Stage B: PU learning on multilingual embeddings (Elkan–Noto)
#
# Input expectations:
# - positives.json / .jsonl : ONLY advertisements (positives)
# - unlabeled.json / .jsonl : mixed pool (ads + non-ads), unlabeled
#
# JSON formats supported:
#   (A) Mapping: {"article_1": "text ...", "article_2": "text ..."}
#   (B) Array of objects: [{"id":"...","text":"..."}, {"id":"...","text":"..."}]
#   (C) JSONL: each line is either {"id":"...","text":"..."} or {"some_id":"text ..."}
#
# Usage examples at bottom.

import argparse
import os
import json
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import gc
from joblib import dump

# ---- Optional GPU embeddings ----
try:
    import torch
    from sentence_transformers import SentenceTransformer
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# ----------------------------
# IO helpers for JSON/JSONL
# ----------------------------
def _read_json_any(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Try JSON first; if it fails, treat as JSONL
    try:
        return json.loads(raw), "json"
    except json.JSONDecodeError:
        # JSONL
        items = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items, "jsonl"

def texts_from_json_like(obj, text_key: str = "text") -> List[str]:
    """
    Accepts:
      - dict mapping id -> text
      - list of dicts with a 'text' field (configurable via text_key)
      - list of dicts mapping single id -> text (JSONL mapping style)
    Returns list[str] of texts.
    """
    texts: List[str] = []
    if isinstance(obj, dict):
        # mapping id -> text
        for _, v in obj.items():
            if isinstance(v, str):
                texts.append(v)
            elif isinstance(v, dict) and text_key in v and isinstance(v[text_key], str):
                texts.append(v[text_key])
    elif isinstance(obj, list):
        for entry in obj:
            if isinstance(entry, dict):
                if text_key in entry and isinstance(entry[text_key], str):
                    texts.append(entry[text_key])
                else:
                    # maybe it's a single-key mapping line: {"article_123": "text ..."}
                    if len(entry) == 1:
                        v = list(entry.values())[0]
                        if isinstance(v, str):
                            texts.append(v)
    return [str(t) if t is not None else "" for t in texts]

def read_json_texts(path: str, text_key: str = "text") -> List[str]:
    obj, kind = _read_json_any(path)
    texts = texts_from_json_like(obj, text_key=text_key)
    if not texts:
        raise ValueError(
            f"No texts parsed from {path}. "
            f"Acceptable formats: mapping id->text, array of objects with '{text_key}', or JSONL with per-line objects."
        )
    return texts

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ----------------------------
# Stage A: TF-IDF + Linear
# ----------------------------
@dataclass
class StageAConfig:
    min_df: int = 3
    max_features: int = 500_000
    keep_rate_target: float = 0.30      # Aim to keep ~30% of corpus
    target_recall: float = 0.99         # ≥99% recall on positives
    model_type: str = "LinearSVC"       # "LinearSVC" or "LogReg"
    c_pos_weight: float = 5.0           # weight positives vs sampled U during training
    calibrate: bool = True              # probabilities for threshold search

def build_tfidf_char(texts: List[str], cfg: StageAConfig) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5),
        lowercase=True, min_df=cfg.min_df, max_features=cfg.max_features,
        norm="l2"
    )
    vec.fit(texts)
    return vec

def train_stage_a(
    pos_texts_train: List[str],
    u_texts_train: List[str],
    pos_texts_val: List[str],
    vec: TfidfVectorizer,
    cfg: StageAConfig
):
    Xp = vec.transform(pos_texts_train)
    Xu = vec.transform(u_texts_train)

    y = np.concatenate([np.ones(Xp.shape[0]), np.zeros(Xu.shape[0])])
    X = sparse.vstack([Xp, Xu])
    sample_weight = np.ones_like(y, dtype=np.float32)
    sample_weight[y == 1] = cfg.c_pos_weight

    if cfg.model_type == "LinearSVC":
        base = LinearSVC(C=1.0, loss="squared_hinge")
        model = CalibratedClassifierCV(base, cv=3, method="sigmoid") if cfg.calibrate else base
    else:
        model = LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0)

    model.fit(X, y, sample_weight=sample_weight)

    Xp_val = vec.transform(pos_texts_val)
    scores_pos = model.predict_proba(Xp_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Xp_val)

    X_keep_probe = Xu[: min(100_000, Xu.shape[0])]
    scores_probe = model.predict_proba(X_keep_probe)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_keep_probe)

    thresholds = np.unique(np.concatenate([scores_pos, scores_probe]))
    thresholds.sort()

    def recall_at(t): return (scores_pos >= t).mean()
    def keep_rate_at(t): return (scores_probe >= t).mean()

    viable = []
    step = max(1, len(thresholds)//5000)
    for t in thresholds[::step]:
        r = recall_at(t)
        if r >= cfg.target_recall:
            k = keep_rate_at(t)
            viable.append((t, r, k))

    if not viable:
        t_star = float(np.percentile(scores_pos, 1))  # keep ~99% positives
    else:
        t_star, r_star, k_star = min(viable, key=lambda z: abs(z[2] - cfg.keep_rate_target))
        print(f"[Stage A] threshold={t_star:.6f} | recall={r_star:.4f} | est_keep_rate={k_star:.4f}")

    return model, float(t_star)

def apply_stage_a(texts: List[str], vec: TfidfVectorizer, model, threshold: float):
    X = vec.transform(texts)
    scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    keep_mask = scores >= threshold
    return keep_mask, scores


# ----------------------------
# Stage B: PU on embeddings
# ----------------------------
@dataclass
class StageBConfig:
    embed_model: str = "intfloat/multilingual-e5-small"
    batch_size: int = 512
    use_half: bool = True
    lr_C: float = 1.0
    max_iter: int = 2000
    calibrate: bool = True

def load_embedder(name: str):
    if not TORCH_OK:
        raise RuntimeError("PyTorch/sentence-transformers not available. Install torch + sentence-transformers.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(name, device=device)
    return model

def embed_texts(embedder, texts: List[str], batch_size: int = 512) -> np.ndarray:
    embs = embedder.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    )
    return embs.astype(np.float32)

def elkan_noto_prior_estimate(s_pos: np.ndarray) -> float:
    c = float(np.clip(s_pos.mean(), 1e-6, 1.0))
    return c

def elkan_noto_posterior(s: np.ndarray, c: float) -> np.ndarray:
    return np.clip(s / max(c, 1e-6), 0.0, 1.0)

def train_stage_b_pu(
    pos_texts: List[str],
    u_texts: List[str],
    pos_texts_val: List[str],
    u_texts_val: List[str],
    cfg: StageBConfig
):
    embedder = load_embedder(cfg.embed_model)
    E_pos = embed_texts(embedder, pos_texts, cfg.batch_size)
    E_u   = embed_texts(embedder, u_texts,   cfg.batch_size)
    E_pos_val = embed_texts(embedder, pos_texts_val, cfg.batch_size)
    E_u_val   = embed_texts(embedder, u_texts_val,   cfg.batch_size)

    X = np.vstack([E_pos, E_u])
    y = np.concatenate([np.ones(E_pos.shape[0]), np.zeros(E_u.shape[0])])

    base = LogisticRegression(max_iter=cfg.max_iter, C=cfg.lr_C, n_jobs=-1)
    clf = CalibratedClassifierCV(base, cv=3, method="sigmoid") if cfg.calibrate else base
    clf.fit(X, y)

    s_pos_val = clf.predict_proba(E_pos_val)[:, 1]
    c = elkan_noto_prior_estimate(s_pos_val)

    s_pos = s_pos_val
    s_u = clf.predict_proba(E_u_val)[:, 1]
    p_pos = elkan_noto_posterior(s_pos, c)
    p_u = elkan_noto_posterior(s_u, c)
    y_val = np.concatenate([np.ones_like(p_pos), np.zeros_like(p_u)])
    scores_val = np.concatenate([p_pos, p_u])

    ap = average_precision_score(y_val, scores_val)
    prec, rec, thr = precision_recall_curve(y_val, scores_val)

    return clf, {"embed_model": cfg.embed_model, "prior_c": c}, {
        "average_precision": float(ap),
        "pr_curve": {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "thresholds": thr.tolist()
        }
    }


# ----------------------------
# Orchestration / CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Two-stage Ads detector (JSON/JSONL input)")
    parser.add_argument("--pos_json", required=True, help="JSON/JSONL file with ONLY ads")
    parser.add_argument("--u_json", required=True, help="JSON/JSONL file with unlabeled (mixed)")
    parser.add_argument("--val_pos_json", default=None, help="Optional held-out positives JSON/JSONL")
    parser.add_argument("--val_u_json", default=None, help="Optional held-out unlabeled JSON/JSONL")
    parser.add_argument("--text_key", default="text", help="Field name for text when using array-of-objects")
    parser.add_argument("--out_dir", default="outputs_ads")
    parser.add_argument("--sample_u_train", type=int, default=200000, help="sample size of U for Stage A training")
    parser.add_argument("--sample_u_val", type=int, default=50000, help="sample size of U for Stage B validation")
    parser.add_argument("--stageA_keep_rate", type=float, default=0.30)
    parser.add_argument("--stageA_recall", type=float, default=0.99)
    parser.add_argument("--embed_model", default="intfloat/multilingual-e5-small")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "models"))
    ensure_dir(os.path.join(args.out_dir, "reports"))

    # Load data (JSON)
    pos_all = read_json_texts(args.pos_json, text_key=args.text_key)
    u_all   = read_json_texts(args.u_json,   text_key=args.text_key)

    # Train/val split (positives) if no held-out provided
    if args.val_pos_json:
        pos_val = read_json_texts(args.val_pos_json, text_key=args.text_key)
        pos_train = pos_all
    else:
        if len(pos_all) < 5:
            raise ValueError("Need at least a handful of positives to split train/val. Provide --val_pos_json otherwise.")
        cut = int(0.8 * len(pos_all))
        pos_train, pos_val = pos_all[:cut], pos_all[cut:]

    # Sample U
    rng = np.random.default_rng(42)
    u_train = list(rng.choice(u_all, size=min(args.sample_u_train, len(u_all)), replace=False))
    if args.val_u_json:
        u_val = read_json_texts(args.val_u_json, text_key=args.text_key)
    else:
        u_val = list(rng.choice(u_all, size=min(args.sample_u_val, len(u_all)), replace=False))

    # ---------------- Stage A ----------------
    a_cfg = StageAConfig(
        keep_rate_target=args.stageA_keep_rate,
        target_recall=args.stageA_recall,
        model_type="LinearSVC",
        calibrate=True
    )
    print("[Stage A] Fitting TF-IDF vectorizer...")
    vec = build_tfidf_char(pos_train + u_train, a_cfg)

    print("[Stage A] Training high-recall classifier...")
    stageA_model, stageA_thresh = train_stage_a(
        pos_train, u_train, pos_val, vec, a_cfg
    )

    dump(vec, os.path.join(args.out_dir, "models", "stageA_vectorizer.joblib"))
    dump(stageA_model, os.path.join(args.out_dir, "models", "stageA_model.joblib"))
    save_json({"threshold": stageA_thresh, "config": a_cfg.__dict__},
              os.path.join(args.out_dir, "reports", "stageA_meta.json"))

    keep_mask_probe, _ = apply_stage_a(u_train, vec, stageA_model, stageA_thresh)
    est_keep_rate = float(keep_mask_probe.mean())
    print(f"[Stage A] Estimated keep-rate on U probe: {est_keep_rate:.4f}")

    # Filter for Stage B
    print("[Stage A] Filtering survivors for Stage B embedding...")
    keep_mask_pos, _ = apply_stage_a(pos_train, vec, stageA_model, stageA_thresh)
    keep_mask_u, _   = apply_stage_a(u_train,  vec, stageA_model, stageA_thresh)
    pos_train_survivors = [t for t, k in zip(pos_train, keep_mask_pos) if k]
    u_train_survivors   = [t for t, k in zip(u_train, keep_mask_u) if k]

    # ---------------- Stage B (PU) ----------------
    b_cfg = StageBConfig(embed_model=args.embed_model)
    print("[Stage B] Training PU classifier on embeddings...")
    clf_b, artifacts_b, report_b = train_stage_b_pu(
        pos_train_survivors, u_train_survivors, pos_val, u_val, b_cfg
    )

    dump(clf_b, os.path.join(args.out_dir, "models", "stageB_clf.joblib"))
    save_json({"artifacts": artifacts_b, "report": report_b, "config": b_cfg.__dict__},
              os.path.join(args.out_dir, "reports", "stageB_report.json"))

    print("[Done] Model artifacts saved to", args.out_dir)
    print("Key metrics (Stage B): AP =", report_b["average_precision"])
    print("\nInference quickstart:")
    print(" 1) Load vectorizer + Stage A model + threshold")
    print(" 2) Keep texts with score >= threshold")
    print(" 3) Embed survivors with", b_cfg.embed_model, "and apply PU posterior via Elkan–Noto (c =", artifacts_b["prior_c"], ")")
    print(" 4) Choose threshold for your desired precision/recall")

if __name__ == "__main__":
    main()
