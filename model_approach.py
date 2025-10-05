# model_approach.py
"""
Usage Examples:
python model_approach.py --in ads_1000.jsonl --out test.jsonl --ad_threshold 0.1

Basic usage:
python model_approach.py --in input.jsonl --out results.jsonl

With custom threshold and batch size:
python model_approach.py --in input.jsonl --out results.jsonl --ad_threshold 0.65 --batch_size 32

With language-specific thresholds:
python model_approach.py --in input.jsonl --out results.jsonl --lang_thresholds "fr:0.58,de:0.62,lb:0.58"

With text chunking and pooling:
python model_approach.py --in input.jsonl --out results.jsonl --chunk_words 200 --pool logits_weighted

With meta-classifier stacking:
python model_approach.py --in input.jsonl --out results.jsonl --meta_clf meta_classifier.pkl

Full example with all options:
python model_approach.py \
    --in input.jsonl \
    --out results.jsonl \
    --model classla/xlm-roberta-base-multilingual-text-genre-classifier \
    --batch_size 16 \
    --max_length 512 \
    --chunk_words 150 \
    --pool logits_weighted \
    --ad_threshold 0.60 \
    --lang_thresholds "fr:0.58,de:0.62,lb:0.58" \
    --short_len 40 \
    --short_bonus 0.20 \
    --min_words 5 \
    --temperature 1.0 \
    --meta_clf meta_classifier.pkl

Input JSONL format: Each line should contain a JSON object with at least an 'ft' field containing the text to classify.
Output: JSONL with additional fields including promotion_prob, is_ad_pred, xgenre_top_label, etc.
"""

import argparse, json, sys, re, math, pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------
# Utilities: IO
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

def chunk_words(text: str, max_words: int):
    if max_words <= 0:
        yield text
        return
    ws = text.split()
    for i in range(0, len(ws), max_words):
        yield " ".join(ws[i:i+max_words])

# ----------------------------
# Text normalization (light OCR clean)
# ----------------------------
NORM_SPACES = re.compile(r"\s{2,}")
def normalize_text(t: str) -> str:
    if not isinstance(t, str):
        t = str(t)
    t = re.sub(r"[_`~^]+", " ", t)
    t = re.sub(r"\s+([.,:;!?()])", r"\1", t)
    t = NORM_SPACES.sub(" ", t)
    return t.strip()

# ----------------------------
# Rules & Features (expanded)
# ----------------------------
PHONE = re.compile(r"(?:\+?\d{2,3}[\s./-]?)?(?:\(?0\d{1,3}\)?[\s./-]?)\d(?:[\d\s./-]{5,})")
PRICE = re.compile(
    r"(?:CHF|SFr\.?|Fr\.?|fr\.?|€|\$)\s?\d{1,3}(?:[’'`\s]?\d{3})*(?:\.-|[.-])?"
    r"|(?:\d{1,3}(?:[’'`\s]?\d{3})*)(?:\s?(?:CHF|SFr\.?|Fr\.?|€|\$))(?:\.-|[.-])?",
    re.I
)
AREA = re.compile(r"\b\d{2,4}\s?m(?:²|2)\b")
ROOMS_FR = re.compile(r"\b(\d{1,2})\s?pi[eè]ce?s?\b", re.I)
KM = re.compile(r"\b\d{1,3}’?\d{3}\s?km\b", re.I)
YEAR = re.compile(r"\b(19|20)\d{2}\b")
ZIP_CH = re.compile(r"\b\d{4}\b")
ADDRESS = re.compile(r"\b(Rue|Av\.?|Avenue|Platz|Str\.?|Strasse|Grand’Rue|Place)\b", re.I)

CUES_FR = r"(?:à\s?vendre|a\s?vendre|à\s?louer|a\s?louer|à\s?remettre|prix\s+à\s+discuter|écrire\s+à|sous\s+chiffres|tél\.?|loyer|charges|villa|attique|expertisée)"
CUES_DE = r"(?:zu\s?verkaufen|zu\s?vermieten|Preis|Schreib(?:en)?\s+an|unter\s+Chiffre|Tel\.?|Miete|Zimmer|Attika|expertisiert)"
CUES_LB = r"(?:ze\s?verkafen|ze\s?verlounen|Präis|Annonce|Tel\.?)"
CUES = re.compile(fr"\b(?:{CUES_FR}|{CUES_DE}|{CUES_LB})\b", re.I)

def rule_flags(t):
    return {
        "has_phone": bool(PHONE.search(t)),
        "has_price": bool(PRICE.search(t)),
        "has_area": bool(AREA.search(t)),
        "has_rooms": bool(ROOMS_FR.search(t)),
        "has_km": bool(KM.search(t)),
        "has_year": bool(YEAR.search(t)),
        "has_zip": bool(ZIP_CH.search(t)),
        "has_address": bool(ADDRESS.search(t)),
        "has_cue": bool(CUES.search(t)),
        "len_words": len(t.split()),
        "pct_digits": (sum(ch.isdigit() for ch in t) / max(len(t),1)),
    }

# ----------------------------
# Thresholding helpers
# ----------------------------
def parse_lang_thresholds(s: str):
    m = {}
    for kv in s.split(","):
        if not kv.strip():
            continue
        lang, thr = kv.split(":")
        m[lang.strip().lower()] = float(thr)
    return m

def lang_len_threshold(lang: str, n_words: int, lang_thr_map, default_thr: float, short_bonus: float, short_len: int):
    base = lang_thr_map.get((lang or "").lower(), default_thr)
    if n_words <= short_len:
        return max(0.0, base - short_bonus)
    return base

# ----------------------------
# Feature vector for optional meta-classifier stacking
# ----------------------------
FEATURE_NAMES = [
    "promotion_prob_calibrated",
    "len_words","pct_digits",
    "has_price","has_phone","has_cue","has_area","has_rooms","has_km","has_year","has_zip","has_address",
    "lg_fr","lg_de","lg_lb","lg_other"
]

def build_features(text_norm: str, lg: str, promo_prob_cal: float):
    f = rule_flags(text_norm)
    lg = (lg or "").lower()
    onehots = {
        "lg_fr": 1.0 if lg=="fr" else 0.0,
        "lg_de": 1.0 if lg=="de" else 0.0,
        "lg_lb": 1.0 if lg in ("lb","lux","lu","lb-lu") else 0.0,
    }
    onehots["lg_other"] = 1.0 - (onehots["lg_fr"]+onehots["lg_de"]+onehots["lg_lb"])
    x = [
        promo_prob_cal,
        float(f["len_words"]), float(f["pct_digits"]),
        float(f["has_price"]), float(f["has_phone"]), float(f["has_cue"]), float(f["has_area"]), float(f["has_rooms"]),
        float(f["has_km"]), float(f["has_year"]), float(f["has_zip"]), float(f["has_address"]),
        onehots["lg_fr"], onehots["lg_de"], onehots["lg_lb"], onehots["lg_other"]
    ]
    return np.array(x, dtype=np.float32), f

# ----------------------------
# Inference
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL (must contain 'ft')")
    ap.add_argument("--out", required=True, help="Output JSONL with scores")
    ap.add_argument("--model", default="classla/xlm-roberta-base-multilingual-text-genre-classifier")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512, help="Max tokens per chunk")
    ap.add_argument("--chunk_words", type=int, default=0, help="0=no chunking; else words per chunk")
    ap.add_argument("--pool", choices=["max","mean","logits_max","logits_mean","logits_weighted"], default="logits_weighted",
                    help="Pooling across chunks; logits_* strongly recommended")
    ap.add_argument("--ad_threshold", type=float, default=0.60, help="Fallback Promotion prob threshold")
    ap.add_argument("--lang_thresholds", type=str, default="", help="Per-language thresholds, e.g. 'fr:0.58,de:0.62,lb:0.58'")
    ap.add_argument("--short_len", type=int, default=40, help="Word count considered 'short'")
    ap.add_argument("--short_bonus", type=float, default=0.20, help="Lower threshold by this for short texts")
    ap.add_argument("--min_words", type=int, default=0, help="Skip docs with fewer words (0=disabled)")
    ap.add_argument("--temperature", type=float, default=1.0, help="Calibration temperature T; divide logits by T")
    ap.add_argument("--meta_clf", type=str, default="", help="Optional path to scikit-learn pickle (expects features FEATURE_NAMES)")
    args = ap.parse_args()

    lang_thr_map = parse_lang_thresholds(args.lang_thresholds) if args.lang_thresholds else {}

    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device).eval()

    id2label = model.config.id2label
    label2id = {v:k for k,v in id2label.items()}
    promo_id = None
    for i,lab in id2label.items():
        if lab.lower() == "promotion":
            promo_id = i; break
    if promo_id is None:
        raise RuntimeError("Could not find 'Promotion' label in model.config.id2label")

    # optional meta-classifier
    meta_clf = None
    if args.meta_clf:
        with open(args.meta_clf, "rb") as f:
            meta_clf = pickle.load(f)

    out_rows = []

    # buffering
    buf_texts, buf_meta, buf_chunk_counts, buf_chunk_lens = [], [], [], []

    def flush_batch():
        nonlocal out_rows, buf_texts, buf_meta, buf_chunk_counts, buf_chunk_lens
        if not buf_texts: return
        enc = tokenizer(
            buf_texts, padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt"
        )
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits  # (n_chunks, n_labels)
        logits_np = logits.cpu().numpy()

        pos = 0
        for meta, n_chunks in zip(buf_meta, buf_chunk_counts):
            L = logits_np[pos:pos+n_chunks]      # (n_chunks, n_labels)
            lens = buf_chunk_lens[pos:pos+n_chunks]
            pos += n_chunks

            # Pooling
            if args.pool == "max":
                pooled_probs = torch.softmax(torch.tensor(L), dim=-1).max(dim=0).values.numpy()
            elif args.pool == "mean":
                pooled_probs = torch.softmax(torch.tensor(L), dim=-1).mean(dim=0).numpy()
            elif args.pool == "logits_max":
                pooled_logits = L.max(axis=0)
                pooled_logits = pooled_logits / max(args.temperature, 1e-6)
                pooled_probs = torch.softmax(torch.tensor(pooled_logits), dim=-1).numpy()
            elif args.pool == "logits_mean":
                pooled_logits = L.mean(axis=0)
                pooled_logits = pooled_logits / max(args.temperature, 1e-6)
                pooled_probs = torch.softmax(torch.tensor(pooled_logits), dim=-1).numpy()
            else:  # logits_weighted (default)
                w = np.array(lens, dtype=np.float32)
                w = w / (w.sum() + 1e-9)
                pooled_logits = (L * w[:, None]).sum(axis=0)
                pooled_logits = pooled_logits / max(args.temperature, 1e-6)
                pooled_probs = torch.softmax(torch.tensor(pooled_logits), dim=-1).numpy()

            promo_prob = float(pooled_probs[promo_id])
            top_id = int(np.argmax(pooled_probs))
            top_label = id2label[top_id]
            top_prob = float(pooled_probs[top_id])

            # Prepare thresholds
            lg = (meta.get("lg") or meta.get("lang") or "").lower()
            text_raw = meta.get("ft","")
            text_norm = normalize_text(text_raw)
            flags = rule_flags(text_norm)
            thr = lang_len_threshold(lg, flags["len_words"], lang_thr_map, args.ad_threshold, args.short_bonus, args.short_len)

            # Optional stacking with meta-classifier
            if meta_clf is not None:
                x, _ = build_features(text_norm, lg, promo_prob)
                # predict_proba for class 1 (ad)
                try:
                    meta_prob = float(meta_clf.predict_proba(x.reshape(1,-1))[0,1])
                except Exception:
                    # fallback to decision_function if needed
                    s = float(meta_clf.decision_function(x.reshape(1,-1))[0])
                    meta_prob = 1 / (1 + math.exp(-s))
                final_prob = 0.5*promo_prob + 0.5*meta_prob  # simple blend; adjust if desired
            else:
                final_prob = promo_prob
            
            # Original rule scoring - keeping it simple
            rule_score = (
                1.0 * float(flags["has_price"])
                + 1.0 * float(flags["has_phone"])
                + 0.7 * float(flags["has_cue"])
                + 0.4 * float(flags["has_area"])
                + 0.4 * float(flags["has_rooms"])
                + 0.3 * float(flags["has_address"])
            )
            rule_hit = rule_score >= 0.5
            
            # Confidence-based approach
            model_confidence = abs(promo_prob - 0.5) * 2  # Scale 0-1, higher means more confident
            
            # Balance rule influence based on model confidence
            if meta_clf is None:
                if rule_hit and final_prob < thr:
                    # Apply gentler boost when model is confident it's not an ad
                    if promo_prob < 0.3 and model_confidence > 0.4:
                        # Model is confident it's not an ad, apply minimal boost
                        boost = min(0.15, (thr - final_prob) * 0.3)
                    else:
                        # Standard boost approach - adjusted to be less aggressive
                        gap = thr - final_prob
                        boost = gap * 0.4  # Less aggressive boost than before
                    
                    final_prob += boost
                    final_prob = min(final_prob, thr * 0.98)  # Don't quite reach threshold
            
            # Only apply promotion label boost when we're already close
            if top_label in ["Promotion"]:
                if final_prob > (thr * 0.8):
                    final_prob = max(final_prob, thr)
                else:
                    final_prob = max(final_prob, final_prob * 1.15)  # Small boost
                    
            is_ad_pred = bool(final_prob >= thr)

            meta_out = dict(meta)
            meta_out["promotion_prob"] = round(promo_prob, 6)
            meta_out["promotion_prob_final"] = round(final_prob, 6)
            meta_out["xgenre_top_label"] = top_label
            meta_out["xgenre_top_prob"] = round(top_prob, 6)
            meta_out["is_ad_pred"] = is_ad_pred
            # attach diagnostics
            meta_out.update(flags)
            meta_out["rule_hit"] = rule_hit
            meta_out["rule_score"] = round(rule_score, 3)
            meta_out["threshold_used"] = thr
            out_rows.append(meta_out)

        # clear
        buf_texts.clear(); buf_meta.clear(); buf_chunk_counts.clear(); buf_chunk_lens.clear()

    # stream docs
    for row in read_jsonl(args.inp):
        txt = row.get("ft", "")
        txt = normalize_text(txt)
        if args.min_words and len(txt.split()) < args.min_words:
            rr = dict(row); rr["promotion_prob"]=None; rr["xgenre_top_label"]=None
            rr["xgenre_top_prob"]=None; rr["is_ad_pred"]=None; rr["note"]="skipped_min_words"
            out_rows.append(rr); continue

        parts = list(chunk_words(txt, args.chunk_words)) if args.chunk_words>0 else [txt]
        lens = [len(p.split()) for p in parts]

        buf_texts.extend(parts)
        buf_meta.append(row)
        buf_chunk_counts.append(len(parts))
        buf_chunk_lens.extend(lens)

        if len(buf_texts) >= args.batch_size:
            flush_batch()
    flush_batch()

    # Write
    write_jsonl(args.out, out_rows)

    # Quick summary to stdout
    promo_count = sum(1 for r in out_rows if r.get("xgenre_top_label") == "Promotion")
    other_count = sum(1 for r in out_rows if r.get("xgenre_top_label") and r.get("xgenre_top_label") != "Promotion")
    final_promo = sum(1 for r in out_rows if r.get("is_ad_pred") is True)
    final_other = sum(1 for r in out_rows if r.get("is_ad_pred") is False)
    
    # Debug: count top categories for non-ads
    non_ad_categories = {}
    for r in out_rows:
        if r.get("is_ad_pred") is False and r.get("xgenre_top_label"):
            label = r.get("xgenre_top_label")
            non_ad_categories[label] = non_ad_categories.get(label, 0) + 1
    
    print(f"Toplabel Promotion: {promo_count} | Toplabel Other: {other_count}")
    print(f"Final is_ad_pred=True: {final_promo} | False: {final_other}")
    print(f"Non-ad categories breakdown: {non_ad_categories}")

if __name__ == "__main__":
    main()
