"""
Shared utilities for the SCEN-EC pipeline.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_codebook(path: str | Path) -> dict[str, Any]:
    """Load the constraint codebook JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sentence_segment(text: str) -> list[str]:
    """
    Simple sentence segmenter. For a rigorous alternative, use nltk or spacy;
    this implementation is deliberately dependency-light for reproducibility.
    """
    text = re.sub(r"\s+", " ", text).strip()
    # Split on . ? ! followed by whitespace and a capital letter or end of string
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in pieces if p.strip()]


def apply_meta_filter(sentence: str, patterns: list[str], min_len: int) -> bool:
    """
    Return True if the sentence PASSES the meta-filter (is retained).
    """
    if len(sentence) < min_len:
        return False
    for pat in patterns:
        if re.search(pat, sentence, flags=re.IGNORECASE):
            return False
    return True


def normalize_row_sum(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize so each row sums to 1 (handles zero-rows by leaving them 0)."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def jensen_shannon(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence between two probability vectors, in nats.
    Returns 0 for identical distributions.
    """
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))


def js_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """
    Convert Jensen-Shannon divergence to a similarity in [0, 1].
    JS divergence between probability vectors in nats has upper bound ln(2).
    """
    d = jensen_shannon(p, q)
    return 1.0 - d / np.log(2)


def ensure_dir(path: str | Path) -> Path:
    """Create the directory if it doesn't exist and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
