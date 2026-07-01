"""
02_build_embeddings.py
----------------------
Compute sentence-transformer embeddings for every sentence in the corpus.

Input : the sentences.csv from step 01
Output:
    - embeddings.npy    (float32 array, shape [n_sentences, 384])
    - embeddings_index.csv (episode_id, sentence_index -> row in the array)

Usage:
    python scripts/02_build_embeddings.py \\
        --input outputs/sentences.csv \\
        --output outputs/embeddings.npy \\
        --index outputs/embeddings_index.csv \\
        --model sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help="Output .npy embeddings")
    ap.add_argument("--index", required=True, help="Output CSV index (episode_id, sentence_index)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"[02_embed] loaded {len(df):,} sentences")

    model = SentenceTransformer(args.model)
    print(f"[02_embed] model: {args.model}, dim = {model.get_sentence_embedding_dimension()}")

    embs = model.encode(
        df["sentence_text"].astype(str).tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embs)

    df[["episode_id", "sentence_index"]].to_csv(args.index, index=False)
    print(f"[02_embed] wrote {embs.shape} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
