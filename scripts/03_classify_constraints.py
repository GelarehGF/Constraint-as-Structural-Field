"""
03_classify_constraints.py
--------------------------
Three-stage constraint classification pipeline:
    Stage 1: meta-filter (length + boilerplate)
    Stage 2: constraint-bearing gate (pattern + embedding fallback)
    Stage 3: category assignment (prototype similarity + keyword boost)

Also applies the three deterministic exclusion rules from the codebook.

Input:
    --sentences      outputs/sentences.csv
    --embeddings     outputs/embeddings.npy
    --index          outputs/embeddings_index.csv
    --codebook       data/constraint_codebook.json

Output:
    - outputs/constraint_reclassification_v2.csv
      Columns: episode_id, sentence_index, sentence_text,
               passed_meta, passed_gate,
               predicted_category, similarity, category_scores,
               applied_exclusion_rule
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils import apply_meta_filter, load_codebook


NEGATION_NEED_PATTERN = re.compile(
    r"\b(don'?t|didn'?t|can'?t|couldn'?t|won'?t|wasn'?t|weren'?t|"
    r"never|no|not|lack(?:ed|ing)?|without|"
    r"had to|need(?:ed)?|require(?:d|ment)?|"
    r"struggle(?:d|s)?|difficult|difficulty|obstacle|barrier|hard to)\b",
    flags=re.IGNORECASE,
)


def stage2_pattern_hit(sentence: str) -> bool:
    """Regex-based constraint-bearing signal."""
    return bool(NEGATION_NEED_PATTERN.search(sentence))


def stage2_embedding_gate(
    emb: np.ndarray, gate_prototype_embs: np.ndarray, threshold: float
) -> bool:
    """Embedding fallback: passes if max similarity to any gate prototype > threshold."""
    sims = gate_prototype_embs @ emb  # emb is normalized
    return bool(sims.max() > threshold)


def stage3_classify(
    emb: np.ndarray,
    sentence: str,
    category_prototype_embs: dict[str, np.ndarray],
    category_keywords: dict[str, set[str]],
    category_multiword: dict[str, list[str]],
    threshold: float,
    keyword_boost: float = 0.05,
) -> tuple[str, float, dict[str, float]]:
    """Return (predicted_category, similarity, category_scores)."""
    sent_lower = sentence.lower()
    scores: dict[str, float] = {}
    for cat, prot_embs in category_prototype_embs.items():
        max_sim = float((prot_embs @ emb).max())
        # Keyword boost
        kw_hit = any(kw in sent_lower for kw in category_keywords[cat])
        mw_hit = any(mw in sent_lower for mw in category_multiword[cat])
        boost = keyword_boost if (kw_hit or mw_hit) else 0.0
        scores[cat] = max_sim + boost
    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]
    if best_score < threshold:
        return ("unclassified", best_score, scores)
    return (best_cat, best_score, scores)


def apply_exclusion_rules(sentence: str, predicted: str) -> tuple[str, str | None]:
    """
    Apply the three deterministic post-classification correction rules.
    Returns (final_category, applied_rule_id_or_None).
    """
    s = sentence.lower()

    # R1: "could not afford to hire" => financial, not talent
    if predicted == "talent" and re.search(
        r"(couldn'?t|could not|no money to|didn'?t have.*money to)\s+.*(hire|recruit|pay)", s
    ):
        return ("financial", "R1")

    # R2: "didn't know how" => capability, not talent
    if predicted == "talent" and re.search(
        r"(didn'?t know how|had to learn|no expertise|taught ourselves)", s
    ):
        return ("capability", "R2")

    # R3: "for ethical reasons" / "against our values" => self_imposed
    if predicted != "self_imposed" and re.search(
        r"(for ethical reasons?|against our (values|principles)|"
        r"we (chose|decided) not to|based on our (values|mission))", s
    ):
        return ("self_imposed", "R3")

    return (predicted, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    codebook = load_codebook(args.codebook)
    sim_threshold = float(codebook["similarity_threshold"])
    gate_threshold = float(codebook["gate_threshold"])
    meta_min_len = int(codebook["meta_filter"]["min_sentence_length_chars"])
    excluded_patterns = codebook["meta_filter"]["excluded_patterns"]

    sents_df = pd.read_csv(args.sentences)
    idx_df = pd.read_csv(args.index)
    embs = np.load(args.embeddings)

    assert len(sents_df) == len(idx_df) == len(embs), \
        "sentences.csv, index, and embeddings must align"

    # Pre-compute prototype embeddings for gate and each category
    model = SentenceTransformer(args.model)
    gate_prototype_embs = model.encode(
        codebook["gate_prototypes_for_embedding_fallback"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    category_prototype_embs: dict[str, np.ndarray] = {}
    category_keywords: dict[str, set[str]] = {}
    category_multiword: dict[str, list[str]] = {}
    for cat, defn in codebook["categories"].items():
        category_prototype_embs[cat] = model.encode(
            defn["prototype_sentences"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        category_keywords[cat] = set(k.lower() for k in defn["keywords"])
        category_multiword[cat] = [p.lower() for p in defn["multiword_phrases"]]

    rows = []
    for i, row in enumerate(sents_df.itertuples(index=False)):
        emb = embs[i]
        sent = str(row.sentence_text)

        passed_meta = apply_meta_filter(sent, excluded_patterns, meta_min_len)

        if not passed_meta:
            rows.append({
                "episode_id": row.episode_id,
                "sentence_index": row.sentence_index,
                "sentence_text": sent,
                "passed_meta": False,
                "passed_gate": False,
                "predicted_category": "filtered_out",
                "similarity": np.nan,
                "applied_exclusion_rule": None,
            })
            continue

        passed_gate = stage2_pattern_hit(sent) or stage2_embedding_gate(
            emb, gate_prototype_embs, gate_threshold
        )

        if not passed_gate:
            rows.append({
                "episode_id": row.episode_id,
                "sentence_index": row.sentence_index,
                "sentence_text": sent,
                "passed_meta": True,
                "passed_gate": False,
                "predicted_category": "non_constraint",
                "similarity": np.nan,
                "applied_exclusion_rule": None,
            })
            continue

        predicted, sim, _scores = stage3_classify(
            emb, sent,
            category_prototype_embs, category_keywords, category_multiword,
            sim_threshold,
        )
        final, rule = apply_exclusion_rules(sent, predicted)

        rows.append({
            "episode_id": row.episode_id,
            "sentence_index": row.sentence_index,
            "sentence_text": sent,
            "passed_meta": True,
            "passed_gate": True,
            "predicted_category": final,
            "similarity": sim,
            "applied_exclusion_rule": rule,
        })

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    counts = out["predicted_category"].value_counts()
    print(f"[03_classify] wrote {len(out):,} rows -> {args.output}")
    print(f"[03_classify] category counts:\n{counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
