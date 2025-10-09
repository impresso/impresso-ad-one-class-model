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
            # Convert numpy types to Python types for JSON serialization
            clean_row = {}
            for k, v in r.items():
                if isinstance(v, (np.float32, np.float64)):
                    clean_row[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)):
                    clean_row[k] = int(v)
                elif isinstance(v, np.bool_):
                    clean_row[k] = bool(v)
                else:
                    clean_row[k] = v
            f.write(json.dumps(clean_row, ensure_ascii=False) + "\n")

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
    r"(?:CHF|SFr\.?|Fr\.?|fr\.?|€|\$|USD|EUR)\s?\d{1,4}(?:[''`\s\.,]?\d{3})*(?:\.-|[.-]|'-)?"
    r"|(?:\d{1,4}(?:[''`\s\.,]?\d{3})*)(?:\s?(?:CHF|SFr\.?|Fr\.?|€|\$|USD|EUR))(?:\.-|[.-]|'-)?",
    re.I
)
AREA = re.compile(r"\b\d{2,4}\s?m(?:²|2)\b")
ROOMS_FR = re.compile(r"\b(\d{1,2})\s?pi[eè]ce?s?\b", re.I)
KM = re.compile(r"\b\d{1,3}’?\d{3}\s?km\b", re.I)
YEAR = re.compile(r"\b(19|20)\d{2}\b")
ZIP_CH = re.compile(r"\b\d{4}\b")
ADDRESS = re.compile(r"\b(Rue|Av\.?|Avenue|Platz|Str\.?|Strasse|Grand’Rue|Place)\b", re.I)

CUES_FR = r"(?:à\s?vendre|a\s?vendre|à\s?louer|a\s?louer|à\s?remettre|prix\s+à\s+discuter|écrire\s+à|sous\s+chiffres|tél\.?|téléphone|loyer|charges|villa|attique|expertisée|contact|offre|demande|urgent|occasion|affaire)"
CUES_DE = r"(?:zu\s?verkaufen|zu\s?vermieten|Preis|Schreib(?:en)?\s+an|unter\s+Chiffre|Tel\.?|Telefon|Miete|Zimmer|Attika|expertisiert|Kontakt|Angebot|dringend|Gelegenheit)"
CUES_LB = r"(?:ze\s?verkafen|ze\s?verlounen|Präis|Annonce|Tel\.?|Telefon|Kontakt)"
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

def calculate_rule_score_and_confidence(flags):
    """Calculate rule score with balanced weights and better confidence measure"""
    # More balanced rule weights - not as aggressive
    rule_score = (
        2.0 * float(flags["has_price"])      # Strong indicator
        + 2.0 * float(flags["has_phone"])    # Strong indicator  
        + 1.5 * float(flags["has_cue"])      # Medium-strong indicator
        + 1.0 * float(flags["has_area"])     # Medium indicator
        + 1.0 * float(flags["has_rooms"])    # Medium indicator
        + 0.8 * float(flags["has_address"])  # Medium indicator
        + 0.5 * float(flags["has_zip"])      # Weak indicator
        + 0.3 * float(flags["has_year"])     # Weak indicator
        + 0.3 * float(flags["has_km"])       # Weak indicator
    )
    
    # Calculate rule confidence based on number and strength of indicators
    strong_indicators = flags["has_price"] + flags["has_phone"]
    medium_indicators = flags["has_cue"] + flags["has_area"] + flags["has_rooms"]
    weak_indicators = flags["has_address"] + flags["has_zip"] + flags["has_year"] + flags["has_km"]
    
    # More conservative rule confidence calculation
    rule_confidence = min(1.0, (strong_indicators * 0.4 + medium_indicators * 0.2 + weak_indicators * 0.1))
    
    return rule_score, rule_confidence

def is_likely_non_ad_content(flags, text_norm, top_label):
    """Enhanced non-ad detection with more patterns"""
    text_lower = text_norm.lower()
    
    # Expanded non-ad indicators
    news_patterns = ["communiqué", "annonce officielle", "avis", "décès", "funeral", "obituary", "mort", "died", "nécrologie"]
    official_patterns = ["commune", "municipalité", "conseil", "administration", "canton", "état", "préfecture", "mairie"]
    event_patterns = ["concert", "festival", "exposition", "conférence", "meeting", "assemblée", "spectacle", "théâtre"]
    legal_patterns = ["ordonnance", "jugement", "tribunal", "procès", "avocat", "notaire", "succession"]
    
    # Check for non-ad patterns
    has_news_pattern = any(pattern in text_lower for pattern in news_patterns)
    has_official_pattern = any(pattern in text_lower for pattern in official_patterns)
    has_event_pattern = any(pattern in text_lower for pattern in event_patterns)
    has_legal_pattern = any(pattern in text_lower for pattern in legal_patterns)
    
    # Non-promotion top labels that should be treated carefully
    careful_labels = ["News", "Obituary", "Official", "Legal", "Academic", "Literature", "Editorial"]
    is_careful_label = top_label in careful_labels
    
    # Very short texts with minimal ad indicators might be false positives
    is_very_short = flags["len_words"] < 12
    has_minimal_indicators = (flags["has_price"] + flags["has_phone"] + flags["has_cue"]) == 0
    
    return {
        "has_news_pattern": has_news_pattern,
        "has_official_pattern": has_official_pattern, 
        "has_event_pattern": has_event_pattern,
        "has_legal_pattern": has_legal_pattern,
        "is_careful_label": is_careful_label,
        "is_suspicious_short": is_very_short and has_minimal_indicators,
        "non_ad_confidence": min(1.0, sum([has_news_pattern, has_official_pattern, has_event_pattern, has_legal_pattern]) * 0.3 + (0.5 if is_careful_label else 0))
    }

def calculate_ensemble_ad_signal(promo_prob, top_label, all_probs, id2label):
    """Calculate additional signal from other genre predictions"""
    # Labels that might indicate commercial content
    commercial_labels = ["Promotion", "Advertisement", "Commercial", "Classified"]
    business_labels = ["Business", "Economic", "Financial"]
    
    commercial_prob = 0.0
    business_prob = 0.0
    
    for idx, prob in enumerate(all_probs):
        label = id2label.get(idx, "")
        if any(cl.lower() in label.lower() for cl in commercial_labels):
            commercial_prob += prob
        elif any(bl.lower() in label.lower() for bl in business_labels):
            business_prob += prob * 0.3  # Weaker signal
    
    # Ensemble signal combining multiple indicators
    ensemble_signal = min(1.0, commercial_prob + business_prob)
    return ensemble_signal

def get_adaptive_threshold_adjustment(flags, text_norm, base_thr):
    """Adjust threshold based on text characteristics"""
    text_lower = text_norm.lower()
    
    # Text length categories
    word_count = flags["len_words"]
    if word_count < 10:
        length_factor = 1.1  # Raise threshold for very short texts
    elif word_count < 25:
        length_factor = 1.05  # Slightly raise for short texts
    elif word_count > 100:
        length_factor = 0.95  # Lower for longer ads
    else:
        length_factor = 1.0
    
    # Digit density - ads often have more numbers
    digit_density = flags["pct_digits"]
    if digit_density > 0.15:
        digit_factor = 0.92  # Lower threshold for number-heavy texts
    elif digit_density > 0.08:
        digit_factor = 0.96
    else:
        digit_factor = 1.0
    
    # Punctuation patterns that suggest ads
    has_contact_pattern = bool(re.search(r"[:\-]\s*\d|contact\s*[:@]|\b\d+[-.\s]\d+[-.\s]\d+", text_lower))
    contact_factor = 0.94 if has_contact_pattern else 1.0
    
    # Multiple currency/price mentions
    price_mentions = len(re.findall(r"(?:CHF|Fr\.?|€|\$|\d+\s*.-)", text_norm, re.I))
    price_factor = max(0.88, 1.0 - (price_mentions * 0.04)) if price_mentions > 1 else 1.0
    
    # Combined adjustment
    adjustment = length_factor * digit_factor * contact_factor * price_factor
    return base_thr * adjustment

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
    # model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_xlm')  # use fine-tuned model
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
            
            # Calculate ensemble signal from all predictions
            ensemble_ad_signal = calculate_ensemble_ad_signal(promo_prob, top_label, pooled_probs, id2label)

            # Prepare thresholds
            lg = (meta.get("lg") or meta.get("lang") or "").lower()
            text_raw = meta.get("ft","")
            text_norm = normalize_text(text_raw)
            flags = rule_flags(text_norm)
            
            # Base threshold with language and length adjustments
            base_thr = lang_len_threshold(lg, flags["len_words"], lang_thr_map, args.ad_threshold, args.short_bonus, args.short_len)
            
            # Apply adaptive threshold adjustment
            thr = get_adaptive_threshold_adjustment(flags, text_norm, base_thr)

            # Optional stacking with meta-classifier
            if meta_clf is not None:
                x, _ = build_features(text_norm, lg, promo_prob)
                try:
                    meta_prob = float(meta_clf.predict_proba(x.reshape(1,-1))[0,1])
                except Exception:
                    s = float(meta_clf.decision_function(x.reshape(1,-1))[0])
                    meta_prob = 1 / (1 + math.exp(-s))
                final_prob = 0.5*promo_prob + 0.5*meta_prob
            else:
                # Blend promotion probability with ensemble signal
                final_prob = promo_prob * 0.85 + ensemble_ad_signal * 0.15
            
            # Enhanced rule scoring with confidence
            rule_score, rule_confidence = calculate_rule_score_and_confidence(flags)
            rule_hit = rule_score >= 1.5
            
            # Check for non-ad patterns to reduce false positives
            non_ad_check = is_likely_non_ad_content(flags, text_norm, top_label)
            
            # Model confidence calculation
            model_confidence = abs(promo_prob - 0.5) * 2
            model_uncertainty = 1.0 - model_confidence
            
            # Rule influence should be inversely related to model confidence
            rule_influence_multiplier = 0.3 + (model_uncertainty * 1.2)
            
            # Enhanced rule-based adjustments with more sophisticated logic
            if meta_clf is None:
                precision_penalty = non_ad_check["non_ad_confidence"] * 0.2
                
                # More nuanced intervention logic
                if model_confidence < 0.75:  # Slightly more permissive threshold
                    
                    # Very strong rule evidence
                    if rule_confidence > 0.7 and rule_score >= 4.0:
                        if precision_penalty < 0.1:
                            base_boost = 0.15 - precision_penalty
                            boost = max(base_boost * rule_influence_multiplier, 0.03)
                            final_prob = max(final_prob, thr + boost)
                        else:
                            boost = max((0.06 - precision_penalty) * rule_influence_multiplier, 0.01)
                            final_prob += boost
                    
                    # Strong rule evidence
                    elif rule_confidence > 0.5 and rule_score >= 3.0:
                        if precision_penalty < 0.1:
                            base_boost = 0.12 - precision_penalty
                            boost = max(base_boost * rule_influence_multiplier, 0.02)
                            final_prob = max(final_prob, thr + boost)
                        else:
                            boost = max((0.05 - precision_penalty) * rule_influence_multiplier, 0.01)
                            final_prob += boost
                    
                    # Medium rule confidence
                    elif rule_confidence > 0.4 and rule_hit:
                        if final_prob < thr and precision_penalty < 0.15:
                            gap = thr - final_prob
                            base_boost = gap * 0.65 + 0.08 - precision_penalty
                            boost = max(base_boost * rule_influence_multiplier, 0.01)
                            final_prob += boost
                        elif precision_penalty < 0.05:
                            boost = 0.04 * rule_influence_multiplier
                            final_prob += boost
                    
                    # Low rule confidence - very selective intervention
                    elif rule_confidence > 0.25 and rule_hit and model_uncertainty > 0.6:
                        if precision_penalty < 0.05:
                            boost = 0.03 * rule_influence_multiplier
                            final_prob += boost
                
                # Enhanced combination bonuses with more patterns
                if model_uncertainty > 0.25:
                    combination_boost = 0.0
                    
                    # Strongest combinations
                    if flags["has_price"] and flags["has_phone"] and precision_penalty < 0.1:
                        combination_boost = (0.16 - precision_penalty) * rule_influence_multiplier
                    elif (flags["has_price"] and flags["has_cue"]) and precision_penalty < 0.1:
                        combination_boost = (0.13 - precision_penalty) * rule_influence_multiplier
                    elif (flags["has_phone"] and flags["has_cue"]) and precision_penalty < 0.1:
                        combination_boost = (0.11 - precision_penalty) * rule_influence_multiplier
                    # New combinations
                    elif flags["has_price"] and flags["has_address"] and flags["has_zip"]:
                        combination_boost = (0.09 - precision_penalty) * rule_influence_multiplier
                    elif flags["has_cue"] and flags["has_area"] and flags["has_rooms"]:
                        combination_boost = (0.08 - precision_penalty) * rule_influence_multiplier
                    
                    if combination_boost > 0:
                        final_prob = max(final_prob, min(final_prob + combination_boost, 0.92))
                
                # Smart threshold adjustment
                if model_uncertainty > 0.35:
                    if rule_score >= 3.5 and precision_penalty < 0.1:
                        threshold_reduction = 0.18 * rule_influence_multiplier
                        thr *= (1.0 - threshold_reduction)
                    elif rule_score >= 2.5 and precision_penalty < 0.05:
                        threshold_reduction = 0.12 * rule_influence_multiplier
                        thr *= (1.0 - threshold_reduction)
                
                # Safety mechanisms
                if model_confidence > 0.85 and promo_prob < 0.15:
                    final_prob *= (0.93 - min(model_confidence * 0.12, 0.18))
                elif model_confidence > 0.85 and promo_prob > 0.85:
                    if precision_penalty > 0:
                        precision_penalty *= (0.4 - model_confidence * 0.25)
                
                # Adaptive precision penalty
                if precision_penalty > 0.15 and promo_prob < 0.35:
                    adjusted_penalty = precision_penalty * (0.4 + model_confidence * 0.6)
                    final_prob *= (0.75 - adjusted_penalty)
            
            # Enhanced genre label handling
            if top_label == "Promotion":
                if rule_hit and precision_penalty < 0.1:
                    base_boost = 0.06 - precision_penalty
                    boost = max(base_boost * (0.6 + model_uncertainty * 0.7), 0.005)
                    final_prob = max(final_prob, thr + boost)
                elif precision_penalty < 0.05:
                    multiplier = 1.12 + (model_uncertainty * 0.18)
                    final_prob = max(final_prob, final_prob * multiplier)
                    
            is_ad_pred = bool(final_prob >= thr)

            meta_out = dict(meta)
            meta_out["promotion_prob"] = round(promo_prob, 6)
            meta_out["promotion_prob_final"] = round(final_prob, 6)
            meta_out["ensemble_ad_signal"] = round(ensemble_ad_signal, 6)
            meta_out["xgenre_top_label"] = top_label
            meta_out["xgenre_top_prob"] = round(top_prob, 6)
            meta_out["is_ad_pred"] = is_ad_pred
            # attach diagnostics
            meta_out.update(flags)
            meta_out.update(non_ad_check)
            meta_out["rule_hit"] = rule_hit
            meta_out["rule_score"] = round(rule_score, 3)
            meta_out["rule_confidence"] = round(rule_confidence, 3)
            meta_out["model_confidence"] = round(model_confidence, 3)
            meta_out["rule_influence_multiplier"] = round(rule_influence_multiplier, 3)
            meta_out["threshold_used"] = round(thr, 6)
            meta_out["base_threshold"] = round(base_thr, 6)
            meta_out["precision_penalty"] = round(precision_penalty, 3)
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
