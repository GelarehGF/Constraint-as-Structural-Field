"""
07_industry_analysis.py
-----------------------
Compute industry-level constraint signatures using lift analysis.

Inputs:
    --profiles episode_level/episode_profiles.csv
    --industries episode_level/episode_industry_labels.csv

Outputs (in --output-dir):
    - Q2_industry_constraint_profile.csv   (industry x constraint mean profile)
    - Q3_lift_industry_x_constraint.csv    (industry x constraint lift matrix)
    - Q3_industry_signatures.csv           (Table 10: over/under-represented per industry)

Usage:
    python scripts/07_industry_analysis.py \\
        --profiles episode_level/episode_profiles.csv \\
        --industries episode_level/episode_industry_labels.csv \\
        --output-dir constraint_x_industry/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORIES = [
    "financial", "market", "institutional", "capability",
    "talent", "network", "self_imposed",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--industries", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-episodes-per-industry", type=int, default=3)
    ap.add_argument("--over-threshold", type=float, default=1.3)
    ap.add_argument("--under-threshold", type=float, default=0.7)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = pd.read_csv(args.profiles)
    industries = pd.read_csv(args.industries)
    df = profiles.merge(industries, on="episode_id", how="inner")

    # Exclude 'unknown'
    df = df[df["industry"] != "unknown"].copy()

    prop_cols = [f"prop_constraint_{c}" for c in CATEGORIES]
    corpus_mean = df[prop_cols].mean()

    industry_means = df.groupby("industry")[prop_cols].mean()
    industry_counts = df.groupby("industry").size().rename("n_episodes")

    # Keep industries with at least min-episodes
    industry_means = industry_means.join(industry_counts)
    industry_means = industry_means[industry_means["n_episodes"] >= args.min_episodes_per_industry]

    # Mean profile matrix (Q2)
    q2 = industry_means[prop_cols].copy()
    q2.columns = CATEGORIES
    q2["n_episodes"] = industry_means["n_episodes"]
    q2.to_csv(out_dir / "Q2_industry_constraint_profile.csv")

    # Lift matrix (Q3)
    lift = industry_means[prop_cols].divide(corpus_mean.values, axis=1)
    lift.columns = CATEGORIES
    lift["n_episodes"] = industry_means["n_episodes"]
    lift = lift.round(2)
    lift.to_csv(out_dir / "Q3_lift_industry_x_constraint.csv")

    # Industry signatures (Table 10)
    rows = []
    for ind, r in lift.iterrows():
        over = [(c, r[c]) for c in CATEGORIES if r[c] >= args.over_threshold]
        under = [(c, r[c]) for c in CATEGORIES if r[c] <= args.under_threshold]
        over_str = "; ".join(f"{c} ({v:.2f})" for c, v in sorted(over, key=lambda x: -x[1]))
        under_str = "; ".join(f"{c} ({v:.2f})" for c, v in sorted(under, key=lambda x: x[1]))
        rows.append({
            "industry": ind,
            "n_episodes": int(r["n_episodes"]),
            "over_represented": over_str,
            "under_represented": under_str,
        })
    sig = pd.DataFrame(rows).sort_values("n_episodes", ascending=False)
    sig.to_csv(out_dir / "Q3_industry_signatures.csv", index=False)

    print(f"[07_industry] wrote Q2/Q3 tables for {len(lift)} industries -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
