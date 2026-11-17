# pipeline.py
import json, re, math, os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from transformers import Pipeline, AutoTokenizer, AutoModelForSequenceClassification

# =========================
# Defaults (from best_params_final.json if present)
# =========================
_DEFAULTS = {
    "pool": "logits_max",
    "temperature": 0.8,
    "short_bonus": 0.2,
    "short_len": 30,
    "min_words": 0,
    "chunk_words": 0,
    "max_length": 512,
    "ad_threshold": 0.961243212223053,
    "lang_thresholds": "other:0.9570,fr:0.8124",
}
# allow overriding by shipping best_params_final.json in the repo
try:
    here = Path(__file__).resolve().parent
    bp = here / "best_params_final.json"
    if bp.exists():
        with open(bp, "r", encoding="utf-8") as f:
            data = json.load(f)
        _DEFAULTS.update({k: v for k, v in data.get("best_params", {}).items() if k in _DEFAULTS})
except Exception:
    pass

# =========================
# Text utils and rules (from model_approach)
# =========================
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
        yield text
        return
    ws = text.split()
    for i in range(0, len(ws), max_words):
        yield " ".join(ws[i:i + max_words])

PHONE = re.compile(r"(?:\+?\d{2,3}[\s./-]?)?(?:\(?0\d{1,3}\)?[\s./-]?)\d(?:[\d\s./-]{5,})")
PRICE = re.compile(
    r"(?:CHF|SFr\.?|Fr\.?|fr\.?|€|\$|USD|EUR)\s?\d{1,4}(?:[''`\s\.,]?\d{3})*(?:\.-|[.-]|'-)?"
    r"|(?:\d{1,4}(?:[''`\s\.,]?\d{3})*)(?:\s?(?:CHF|SFr\.?|Fr\.?|€|\$|USD|EUR))(?:\.-|[.-]|'-)?",
    re.I,
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

def rule_flags(t: str) -> Dict[str, Any]:
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
        "pct_digits": (sum(ch.isdigit() for ch in t) / max(len(t), 1)),
    }

def calculate_rule_score_and_confidence(flags: Dict[str, Any]) -> Tuple[float, float]:
    rule_score = (
        2.0 * float(flags["has_price"])
        + 2.0 * float(flags["has_phone"])
        + 1.5 * float(flags["has_cue"])
        + 1.0 * float(flags["has_area"])
        + 1.0 * float(flags["has_rooms"])
        + 0.8 * float(flags["has_address"])
        + 0.5 * float(flags["has_zip"])
        + 0.3 * float(flags["has_year"])
        + 0.3 * float(flags["has_km"])
    )
    strong = flags["has_price"] + flags["has_phone"]
    med = flags["has_cue"] + flags["has_area"] + flags["has_rooms"]
    weak = flags["has_address"] + flags["has_zip"] + flags["has_year"] + flags["has_km"]
    rule_confidence = min(1.0, (strong * 0.4 + med * 0.2 + weak * 0.1))
    return rule_score, rule_confidence

def is_likely_non_ad_content(flags: Dict[str, Any], text_norm: str, top_label: str) -> Dict[str, Any]:
    tl = text_norm.lower()
    news = ["communiqué", "annonce officielle", "avis", "décès", "funeral", "obituary", "mort", "died", "nécrologie"]
    official = ["commune", "municipalité", "conseil", "administration", "canton", "état", "préfecture", "mairie"]
    event = ["concert", "festival", "exposition", "conférence", "meeting", "assemblée", "spectacle", "théâtre"]
    legal = ["ordonnance", "jugement", "tribunal", "procès", "avocat", "notaire", "succession"]

    has_news = any(p in tl for p in news)
    has_official = any(p in tl for p in official)
    has_event = any(p in tl for p in event)
    has_legal = any(p in tl for p in legal)

    careful_labels = ["News", "Obituary", "Official", "Legal", "Academic", "Literature", "Editorial"]
    is_careful = top_label in careful_labels

    is_short = flags["len_words"] < 12
    minimal = (flags["has_price"] + flags["has_phone"] + flags["has_cue"]) == 0

    return {
        "has_news_pattern": has_news,
        "has_official_pattern": has_official,
        "has_event_pattern": has_event,
        "has_legal_pattern": has_legal,
        "is_careful_label": is_careful,
        "is_suspicious_short": is_short and minimal,
        "non_ad_confidence": min(1.0, (has_news + has_official + has_event + has_legal) * 0.3 + (0.5 if is_careful else 0)),
    }

def calculate_ensemble_ad_signal(promo_prob: float, top_label: str, all_probs: np.ndarray, id2label: Dict[int, str]) -> float:
    commercial_labels = ["Promotion", "Advertisement", "Commercial", "Classified"]
    business_labels = ["Business", "Economic", "Financial"]
    commercial_prob = 0.0
    business_prob = 0.0
    for idx, prob in enumerate(all_probs):
        label = id2label.get(idx, "")
        if any(cl.lower() in label.lower() for cl in commercial_labels):
            commercial_prob += prob
        elif any(bl.lower() in label.lower() for bl in business_labels):
            business_prob += prob * 0.3
    return min(1.0, commercial_prob + business_prob)

def get_adaptive_threshold_adjustment(flags: Dict[str, Any], text_norm: str, base_thr: float) -> float:
    tl = text_norm.lower()
    wc = flags["len_words"]
    if wc < 10:
        length_factor = 1.1
    elif wc < 25:
        length_factor = 1.05
    elif wc > 100:
        length_factor = 0.95
    else:
        length_factor = 1.0

    dd = flags["pct_digits"]
    if dd > 0.15: digit_factor = 0.92
    elif dd > 0.08: digit_factor = 0.96
    else: digit_factor = 1.0

    has_contact = bool(re.search(r"[:\-]\s*\d|contact\s*[:@]|\b\d+[-.\s]\d+[-.\s]\d+", tl))
    contact_factor = 0.94 if has_contact else 1.0

    price_mentions = len(re.findall(r"(?:CHF|Fr\.?|€|\$|\d+\s*.-)", text_norm, re.I))
    price_factor = max(0.88, 1.0 - (price_mentions * 0.04)) if price_mentions > 1 else 1.0

    return base_thr * (length_factor * digit_factor * contact_factor * price_factor)

def parse_lang_thresholds(s: str) -> Dict[str, float]:
    m: Dict[str, float] = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        lang, thr = kv.split(":")
        m[lang.strip().lower()] = float(thr)
    return m

def lang_len_threshold(lang: str, n_words: int, lang_thr_map: Dict[str, float], default_thr: float, short_bonus: float, short_len: int):
    base = lang_thr_map.get((lang or "").lower(), default_thr)
    if n_words <= short_len:
        return max(0.0, base - short_bonus)
    return base

# optional meta-stacking features (kept for parity; only used if user provides a pickle)
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
        "lg_fr": 1.0 if lg == "fr" else 0.0,
        "lg_de": 1.0 if lg == "de" else 0.0,
        "lg_lb": 1.0 if lg in ("lb", "lux", "lu", "lb-lu") else 0.0,
    }
    onehots["lg_other"] = 1.0 - (onehots["lg_fr"] + onehots["lg_de"] + onehots["lg_lb"])
    x = [
        promo_prob_cal,
        float(f["len_words"]), float(f["pct_digits"]),
        float(f["has_price"]), float(f["has_phone"]), float(f["has_cue"]), float(f["has_area"]), float(f["has_rooms"]),
        float(f["has_km"]), float(f["has_year"]), float(f["has_zip"]), float(f["has_address"]),
        onehots["lg_fr"], onehots["lg_de"], onehots["lg_lb"], onehots["lg_other"],
    ]
    return np.array(x, dtype=np.float32), f

# =========================
# The actual pipeline
# =========================
class AdDetectionPipeline(Pipeline):
    task = "text-classification"

    def __init__(self, model=None, tokenizer=None, **kwargs):
        if model is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                kwargs.pop("model_name_or_path", None) or kwargs.pop("pretrained_model_name_or_path", None) or "."
            )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(getattr(model, "name_or_path", "."))
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

        # label mapping
        self.id2label: Dict[int, str] = {int(k): v for k, v in self.model.config.id2label.items()}
        self.promo_id = None
        for i, lab in self.id2label.items():
            if str(lab).lower() == "promotion":
                self.promo_id = i
                break
        if self.promo_id is None:
            raise RuntimeError("Could not find 'Promotion' label in model.config.id2label")

    # let users tweak knobs at call-time
    def _sanitize_parameters(
        self,
        ad_threshold: float = _DEFAULTS["ad_threshold"],
        lang_thresholds: str = _DEFAULTS["lang_thresholds"],
        short_len: int = _DEFAULTS["short_len"],
        short_bonus: float = _DEFAULTS["short_bonus"],
        min_words: int = _DEFAULTS["min_words"],
        chunk_words: int = _DEFAULTS["chunk_words"],
        max_length: int = _DEFAULTS["max_length"],
        pool: str = _DEFAULTS["pool"],
        temperature: float = _DEFAULTS["temperature"],
        meta_clf: str = "",  # optional path within repo
        return_diagnostics: bool = True,
    ):
        pre = dict(chunk_words=chunk_words, max_length=max_length, min_words=min_words)
        fwd = dict(temperature=temperature)
        post = dict(
            ad_threshold=ad_threshold, lang_thresholds=lang_thresholds,
            short_len=short_len, short_bonus=short_bonus,
            pool=pool, meta_clf=meta_clf, return_diagnostics=return_diagnostics
        )
        return pre, fwd, post

    # accept str / dict / list
    def preprocess(self, inputs, chunk_words: int, max_length: int, min_words: int):
        if isinstance(inputs, (str, dict)):
            items = [inputs]
        else:
            items = list(inputs)

        texts, metas, chunk_lens = [], [], []
        for x in items:
            row = {"ft": x} if isinstance(x, str) else x
            t = normalize_text(row.get("ft", ""))
            if min_words and len(t.split()) < min_words:
                metas.append({"_skip": True, **row})
                continue
            parts = list(chunk_words_fn(t, chunk_words)) if chunk_words > 0 else [t]
            texts += parts
            metas.append({"_n": len(parts), "_lens": [len(p.split()) for p in parts], **row})

        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return {"enc": enc, "metas": metas}

    def _forward(self, model_inputs, temperature: float):
        enc = {k: v.to(self.model.device) for k, v in model_inputs["enc"].items()}
        with torch.no_grad():
            logits = self.model(**enc).logits / max(temperature, 1e-6)
        return {"logits": logits.cpu(), "metas": model_inputs["metas"]}

    def postprocess(
        self, model_outputs, ad_threshold: float, lang_thresholds: str,
        short_len: int, short_bonus: float, pool: str, meta_clf: str, return_diagnostics: bool
    ):
        L = model_outputs["logits"].numpy()
        metas = model_outputs["metas"]
        results = []

        # language thresholds
        thr_map = parse_lang_thresholds(lang_thresholds) if lang_thresholds else {}

        # optional meta-classifier
        clf = None
        if meta_clf:
            try:
                p = Path(getattr(self.model, "name_or_path", ".")) / meta_clf
                if p.exists():
                    import pickle
                    with open(p, "rb") as f:
                        clf = pickle.load(f)
            except Exception:
                clf = None

        pos = 0
        for m in metas:
            if m.get("_skip"):
                results.append({"label": "SKIPPED", "score": None, **m})
                continue

            n = m["_n"]; lens = m["_lens"]
            logits_doc = L[pos:pos + n]; pos += n

            # pooling (logits_* recommended)
            if pool == "max":
                pooled_probs = torch.softmax(torch.tensor(logits_doc), dim=-1).max(dim=0).values.numpy()
            elif pool == "mean":
                pooled_probs = torch.softmax(torch.tensor(logits_doc), dim=-1).mean(dim=0).numpy()
            elif pool == "logits_mean":
                pooled = logits_doc.mean(axis=0)
                pooled_probs = torch.softmax(torch.tensor(pooled), dim=-1).numpy()
            elif pool == "logits_weighted":
                w = np.array(lens, dtype=np.float32); w = w / (w.sum() + 1e-9)
                pooled = (logits_doc * w[:, None]).sum(axis=0)
                pooled_probs = torch.softmax(torch.tensor(pooled), dim=-1).numpy()
            else:  # logits_max (default via _DEFAULTS)
                pooled = logits_doc.max(axis=0)
                pooled_probs = torch.softmax(torch.tensor(pooled), dim=-1).numpy()

            promo_prob = float(pooled_probs[self.promo_id])
            top_id = int(np.argmax(pooled_probs))
            top_label = self.id2label[top_id]
            top_prob = float(pooled_probs[top_id])

            # thresholds
            lg = (m.get("lg") or m.get("lang") or "").lower()
            txt = normalize_text(m.get("ft", ""))
            flags = rule_flags(txt)
            base_thr = lang_len_threshold(lg, flags["len_words"], thr_map, ad_threshold, short_bonus, short_len)
            thr = get_adaptive_threshold_adjustment(flags, txt, base_thr)

            # ensemble
            ensemble = calculate_ensemble_ad_signal(promo_prob, top_label, pooled_probs, self.id2label)

            # meta stacking or blend
            if clf is not None:
                x, _ = build_features(txt, lg, promo_prob)
                try:
                    meta_prob = float(clf.predict_proba(x.reshape(1, -1))[0, 1])
                except Exception:
                    s = float(clf.decision_function(x.reshape(1, -1))[0])
                    meta_prob = 1 / (1 + math.exp(-s))
                final_prob = 0.5 * promo_prob + 0.5 * meta_prob
            else:
                final_prob = promo_prob * 0.85 + ensemble * 0.15

            # rules & adjustments (as in model_approach)
            rule_score, rule_conf = calculate_rule_score_and_confidence(flags)
            rule_hit = rule_score >= 1.5
            non_ad = is_likely_non_ad_content(flags, txt, top_label)
            model_conf = abs(promo_prob - 0.5) * 2
            model_unc = 1.0 - model_conf
            rule_mult = 0.3 + (model_unc * 1.2)
            precision_penalty = non_ad["non_ad_confidence"] * 0.2 if clf is None else 0.0

            if clf is None:
                if model_conf < 0.75:
                    if rule_conf > 0.7 and rule_score >= 4.0:
                        if precision_penalty < 0.1:
                            base_boost = 0.15 - precision_penalty
                            boost = max(base_boost * rule_mult, 0.03)
                            final_prob = max(final_prob, thr + boost)
                        else:
                            boost = max((0.06 - precision_penalty) * rule_mult, 0.01)
                            final_prob += boost
                    elif rule_conf > 0.5 and rule_score >= 3.0:
                        if precision_penalty < 0.1:
                            base_boost = 0.12 - precision_penalty
                            boost = max(base_boost * rule_mult, 0.02)
                            final_prob = max(final_prob, thr + boost)
                        else:
                            boost = max((0.05 - precision_penalty) * rule_mult, 0.01)
                            final_prob += boost
                    elif rule_conf > 0.4 and rule_hit:
                        if final_prob < thr and precision_penalty < 0.15:
                            gap = thr - final_prob
                            base_boost = gap * 0.65 + 0.08 - precision_penalty
                            boost = max(base_boost * rule_mult, 0.01)
                            final_prob += boost
                        elif precision_penalty < 0.05:
                            final_prob += 0.04 * rule_mult
                    elif rule_conf > 0.25 and rule_hit and model_unc > 0.6:
                        if precision_penalty < 0.05:
                            final_prob += 0.03 * rule_mult

                if model_unc > 0.25:
                    combo = 0.0
                    if flags["has_price"] and flags["has_phone"] and precision_penalty < 0.1:
                        combo = (0.16 - precision_penalty) * rule_mult
                    elif (flags["has_price"] and flags["has_cue"]) and precision_penalty < 0.1:
                        combo = (0.13 - precision_penalty) * rule_mult
                    elif (flags["has_phone"] and flags["has_cue"]) and precision_penalty < 0.1:
                        combo = (0.11 - precision_penalty) * rule_mult
                    elif flags["has_price"] and flags["has_address"] and flags["has_zip"]:
                        combo = (0.09 - precision_penalty) * rule_mult
                    elif flags["has_cue"] and flags["has_area"] and flags["has_rooms"]:
                        combo = (0.08 - precision_penalty) * rule_mult
                    if combo > 0:
                        final_prob = max(final_prob, min(final_prob + combo, 0.92))

                if model_unc > 0.35:
                    if rule_score >= 3.5 and precision_penalty < 0.1:
                        thr *= (1.0 - 0.18 * rule_mult)
                    elif rule_score >= 2.5 and precision_penalty < 0.05:
                        thr *= (1.0 - 0.12 * rule_mult)

                if model_conf > 0.85 and promo_prob < 0.15:
                    final_prob *= (0.93 - min(model_conf * 0.12, 0.18))
                elif model_conf > 0.85 and promo_prob > 0.85 and precision_penalty > 0:
                    precision_penalty *= (0.4 - model_conf * 0.25)

                if precision_penalty > 0.15 and promo_prob < 0.35:
                    adj = precision_penalty * (0.4 + model_conf * 0.6)
                    final_prob *= (0.75 - adj)

            if top_label == "Promotion":
                if rule_hit and precision_penalty < 0.1:
                    base_boost = 0.06 - precision_penalty
                    boost = max(base_boost * (0.6 + model_unc * 0.7), 0.005)
                    final_prob = max(final_prob, thr + boost)
                elif precision_penalty < 0.05:
                    mult = 1.12 + (model_unc * 0.18)
                    final_prob = max(final_prob, final_prob * mult)

            is_ad = bool(final_prob >= thr)

            out = {
                "label": "AD" if is_ad else "NOT_AD",
                "score": float(final_prob),
                "promotion_prob": float(promo_prob),
                "threshold_used": float(thr),
                "xgenre_top_label": top_label,
                "xgenre_top_prob": top_prob,
            }
            if return_diagnostics:
                out.update({
                    **flags,
                    **non_ad,
                    "rule_hit": rule_hit,
                    "rule_score": round(rule_score, 3),
                    "rule_confidence": round(rule_conf, 3),
                    "model_confidence": round(model_conf, 3),
                })
            results.append(out)
        return results

    # -------------------------
    # Convenience helpers
    # -------------------------
    def predict_jsonl(self, input_path: str, output_path: str, **kwargs):
        """Replicates your JSONL -> JSONL flow from model_approach.py."""
        rows = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        preds = self.__call__(rows, **kwargs)
        # merge diagnostics back into original rows
        out = []
        for r, p in zip(rows, preds):
            rr = dict(r)
            rr["is_ad_pred"] = (p["label"] == "AD")
            rr["promotion_prob"] = p.get("promotion_prob")
            rr["promotion_prob_final"] = p.get("score")
            rr["xgenre_top_label"] = p.get("xgenre_top_label")
            rr["xgenre_top_prob"] = p.get("xgenre_top_prob")
            rr["threshold_used"] = p.get("threshold_used")
            out.append(rr)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return output_path

# tiny alias so AutoPipelineForTextClassification works
def pipeline(*args, **kwargs):
    return AdDetectionPipeline(*args, **kwargs)

def chunk_words_fn(text, max_words):
    return chunk_words(text, max_words)

# Optional: quick CLI for local JSONL processing
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL (must contain 'ft')")
    ap.add_argument("--out", required=True, help="Output JSONL")
    ap.add_argument("--ad_threshold", type=float, default=_DEFAULTS["ad_threshold"])
    ap.add_argument("--lang_thresholds", type=str, default=_DEFAULTS["lang_thresholds"])
    ap.add_argument("--short_len", type=int, default=_DEFAULTS["short_len"])
    ap.add_argument("--short_bonus", type=float, default=_DEFAULTS["short_bonus"])
    ap.add_argument("--min_words", type=int, default=_DEFAULTS["min_words"])
    ap.add_argument("--chunk_words", type=int, default=_DEFAULTS["chunk_words"])
    ap.add_argument("--max_length", type=int, default=_DEFAULTS["max_length"])
    ap.add_argument("--pool", type=str, default=_DEFAULTS["pool"])
    ap.add_argument("--temperature", type=float, default=_DEFAULTS["temperature"])
    ap.add_argument("--meta_clf", type=str, default="")
    args = ap.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(".")
    tok = AutoTokenizer.from_pretrained(".")
    pipe = AdDetectionPipeline(model=model, tokenizer=tok)
    pipe.predict_jsonl(
        args.inp, args.out,
        ad_threshold=args.ad_threshold,
        lang_thresholds=args.lang_thresholds,
        short_len=args.short_len,
        short_bonus=args.short_bonus,
        min_words=args.min_words,
        chunk_words=args.chunk_words,
        max_length=args.max_length,
        pool=args.pool,
        temperature=args.temperature,
        meta_clf=args.meta_clf,
    )
