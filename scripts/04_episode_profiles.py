"""
04_episode_profiles.py
----------------------
Aggregate sentence-level classifications into episode-level compositional profiles.
Produces a 111-episode x 7-category matrix of proportions.

Input : outputs/constraint_reclassification_v2.csv
Output: episode_level/episode_profiles.csv
        with one row per episode and columns:
          episode_id, n_constraint_sentences,
          prop_constraint_financial, prop_constraint_market,
          prop_constraint_institutional, prop_constraint_capability,
          prop_constraint_talent, prop_constraint_network, prop_constraint_self_imposed

Usage:
    python scripts/04_episode_profiles.py \\
        --input outputs/constraint_reclassification_v2.csv \\
        --output episode_level/episode_profiles.csv \\
        --min-constraint-sentences 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CATEGORIES = [
    "financial", "market", "institutional", "capability",
    "talent", "network", "self_imposed",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-constraint-sentences", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["predicted_category"].isin(CATEGORIES)]  # keep only constraint-classified

    profiles = []
    for ep_id, grp in df.groupby("episode_id"):
        n = len(grp)
        if n < args.min_constraint_sentences:
            continue
        row = {"episode_id": ep_id, "n_constraint_sentences": n}
        counts = grp["predicted_category"].value_counts(normalize=True)
        for cat in CATEGORIES:
            row[f"prop_constraint_{cat}"] = float(counts.get(cat, 0.0))
        profiles.append(row)

    out = pd.DataFrame(profiles)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[04_profiles] {len(out)} analyzable episodes "
          f"(threshold: n >= {args.min_constraint_sentences} constraint sentences) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
