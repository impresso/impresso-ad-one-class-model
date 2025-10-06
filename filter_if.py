#!/usr/bin/env python3
"""
Use a trained Isolation Forest pipeline to filter a mixed batch.

Input JSONL (one per line):
  {"id": "...", "lg": "fr", "ft": "full text ..."}

Outputs:
  - Non-Ads JSONL (below threshold)
  - Ads/Remaining JSONL (score >= threshold)
Optionally writes an 'uncertain' file for a small band around the threshold.

Usage:
  python filter_if.py --mix mixed.jsonl --artifacts artifacts \
    --out_nonads nonads.jsonl --out_ads ads.jsonl \
    --uncertain_band 0.01 --out_uncertain uncertain.jsonl
"""
import argparse, json
from pathlib import Path

import numpy as np
import joblib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# (keep in sync with train_if.py)
import unicodedata, re
_WS = re.compile(r"\s+")
_HYPH = re.compile(r"-\s*\n")
def clean_text(txt: str) -> str:
    txt = txt.replace("\r\n", "\n")
    txt = _HYPH.sub("", txt)
    txt = unicodedata.normalize("NFKC", txt)
    txt = _WS.sub(" ", txt).strip()
    return txt

def chunk_words(text: str, max_words: int = 180):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

# ---------- cheap auxiliary features (must match train_if.py exactly) ----------
_CURRENCY_RE = re.compile(r"(?:CHF|Fr\.|€|\$)")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}")
_CUE_RE = re.compile(
    r"\b("
    r"vendre|acheter|louer|remettre|annonce|"
    r"verkauf|verkaufen|mieten|vermieten|gesucht|anzeige|"
    r"t[eé]l\.?|telefon|tel\.?"
    r")\b",
    re.IGNORECASE,
)

def aux_feats(txt: str) -> np.ndarray:
    n_chars = max(len(txt), 1)
    digit_ratio = sum(c.isdigit() for c in txt) / n_chars
    upper_ratio = sum(c.isupper() for c in txt) / n_chars
    token_count = len(txt.split())
    currency = 1.0 if _CURRENCY_RE.search(txt) else 0.0
    phone = 1.0 if _PHONE_RE.search(txt) else 0.0
    cue = 1.0 if _CUE_RE.search(txt) else 0.0
    return np.array([digit_ratio, upper_ratio, np.log1p(token_count), currency, phone, cue], dtype=np.float32)

def read_jsonl(p: str):
    path = Path(p)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(p: str, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def embed_docs(model, docs, chunk_words_n=180, pool="mean", attach_aux=False):
    embs = []
    for d in tqdm(docs, desc="Embedding", unit="doc"):
        raw = d["ft"]
        t = clean_text(raw)
        parts = list(chunk_words(t, max_words=chunk_words_n)) or [t]
        vecs = model.encode(parts, normalize_embeddings=True)
        vecs = np.vstack(vecs) if isinstance(vecs, list) else np.asarray(vecs)
        if pool == "mean":
            v = vecs.mean(axis=0)
        else:
            v = vecs.max(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        if attach_aux:
            feats = aux_feats(t)
            v = np.concatenate([v, feats], dtype=np.float32)
        embs.append(v)
    return np.vstack(embs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True, help="JSONL with mixed/unknown items")
    ap.add_argument("--artifacts", required=True, help="Directory with if_pipeline.joblib + thresholds.json")
    ap.add_argument("--out_nonads", required=True, help="Output JSONL for predicted Non-Ads")
    ap.add_argument("--out_ads", required=True, help="Output JSONL for predicted Ads/Remaining")
    ap.add_argument("--uncertain_band", type=float, default=0.0,
                    help="Optional margin around threshold (absolute score) to route as 'uncertain'")
    ap.add_argument("--out_uncertain", default=None, help="Optional JSONL for uncertain items")
    args = ap.parse_args()

    art_dir = Path(args.artifacts)
    payload = joblib.load(art_dir / "if_pipeline.joblib")
    thresholds = json.loads((art_dir / "thresholds.json").read_text(encoding="utf-8"))

    # Load embedder and params used during training
    embedder = SentenceTransformer(payload["embedder_name"], trust_remote_code=True)
    chunk_words_n = payload.get("chunk_words", 180)
    pool = payload.get("pool", "max")
    attach_aux = bool(payload.get("uses_aux_feats", False))

    # Read & embed
    mix = list(read_jsonl(args.mix))
    if not mix:
        print("No items in mixed file.")
        write_jsonl(args.out_nonads, [])
        write_jsonl(args.out_ads, [])
        return

    X = embed_docs(embedder, mix, chunk_words_n=chunk_words_n, pool=pool, attach_aux=attach_aux)

    # Transform & score
    scaler = payload["scaler"]
    pca = payload["pca"]
    iso = payload["iso"]

    Xc = scaler.transform(X)
    Xp = pca.transform(Xc) if pca is not None else Xc
    scores = iso.decision_function(Xp)  # higher = more Ad-like (inlier)

    # Split by thresholds
    nonads, ads, uncertain = [], [], []
    for item, s in zip(mix, scores):
        lang = item.get("lg", "GLOBAL")
        th = thresholds.get(lang, thresholds.get("GLOBAL", np.quantile(scores, 0.05)))
        # Route with optional uncertainty band
        if args.uncertain_band > 0 and (th - args.uncertain_band) <= s < (th + args.uncertain_band):
            uncertain.append({**item, "if_score": float(s), "threshold": float(th)})
        elif s < th:
            nonads.append({**item, "if_score": float(s), "threshold": float(th)})
        else:
            ads.append({**item, "if_score": float(s), "threshold": float(th)})

    write_jsonl(args.out_nonads, nonads)
    write_jsonl(args.out_ads, ads)
    if args.out_uncertain:
        write_jsonl(args.out_uncertain, uncertain)

    print(f"Predicted Non-Ads: {len(nonads)}")
    print(f"Predicted Ads/Remaining: {len(ads)}")
    if args.out_uncertain:
        print(f"Uncertain (within ±{args.uncertain_band} of threshold): {len(uncertain)}")

if __name__ == "__main__":
    main()
