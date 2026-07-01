"""
09_validate_kappa.py
--------------------
Compute Cohen's kappa for the validation sample:
    - Coder 1 vs Coder 2 (inter-coder / codebook discriminability)
    - Coder 1 vs Pipeline
    - Coder 2 vs Pipeline

Also computes per-category kappa (Appendix C) and confusion matrices (Appendix D).

Inputs:
    --coder1  validation/validation_sample_CODER1_completed.csv
    --coder2  validation/validation_sample_CODER2_completed.csv
    --master  validation/validation_sample.csv  (contains pipeline labels)

Outputs (in --output-dir):
    - kappa_summary.csv                (Table 2)
    - kappa_per_category.csv           (Appendix C)
    - confusion_coder1_vs_coder2.csv
    - confusion_coder1_vs_pipeline.csv
    - confusion_coder2_vs_pipeline.csv
    - disagreements_for_adjudication.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


CATEGORIES_WITH_UNCLASSIFIED = [
    "financial", "market", "institutional", "capability",
    "talent", "network", "self_imposed", "unclassified",
]


def per_category_kappa(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES_WITH_UNCLASSIFIED:
        a_bin = (a == cat).astype(int)
        b_bin = (b == cat).astype(int)
        # cohen_kappa handles the case where a category isn't present
        if a_bin.sum() == 0 and b_bin.sum() == 0:
            k = np.nan
        else:
            k = cohen_kappa_score(a_bin, b_bin)
        rows.append({"category": cat, "kappa": round(float(k), 3) if not np.isnan(k) else np.nan,
                     "n_a": int(a_bin.sum()), "n_b": int(b_bin.sum())})
    return pd.DataFrame(rows)


def confusion_df(a: pd.Series, b: pd.Series, name_a: str, name_b: str) -> pd.DataFrame:
    cm = confusion_matrix(a, b, labels=CATEGORIES_WITH_UNCLASSIFIED)
    df = pd.DataFrame(cm, index=[f"{name_a}:{c}" for c in CATEGORIES_WITH_UNCLASSIFIED],
                      columns=[f"{name_b}:{c}" for c in CATEGORIES_WITH_UNCLASSIFIED])
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coder1", required=True)
    ap.add_argument("--coder2", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c1 = pd.read_csv(args.coder1)
    c2 = pd.read_csv(args.coder2)
    master = pd.read_csv(args.master)

    # Merge on sentence_id (or index if not present)
    key = "sentence_id" if "sentence_id" in c1.columns else "row_id"
    if key not in c1.columns:
        c1 = c1.reset_index().rename(columns={"index": "row_id"})
        c2 = c2.reset_index().rename(columns={"index": "row_id"})
        master = master.reset_index().rename(columns={"index": "row_id"})
        key = "row_id"

    merged = master.merge(
        c1[[key, "coder_label"]].rename(columns={"coder_label": "coder1_label"}),
        on=key,
    ).merge(
        c2[[key, "coder_label"]].rename(columns={"coder_label": "coder2_label"}),
        on=key,
    )

    a = merged["coder1_label"].astype(str)
    b = merged["coder2_label"].astype(str)
    p = merged["pipeline_label"].astype(str) if "pipeline_label" in merged.columns else merged["predicted_category"].astype(str)

    k12 = cohen_kappa_score(a, b)
    k1p = cohen_kappa_score(a, p)
    k2p = cohen_kappa_score(b, p)

    summary = pd.DataFrame([
        {"comparison": "Coder 1 vs Coder 2 (inter-coder)",
         "kappa": round(k12, 3), "n": len(merged),
         "raw_agreement_pct": round(100 * (a == b).mean(), 1),
         "interpretation": "Almost perfect" if k12 > 0.80 else ("Substantial" if k12 > 0.60 else "Moderate")},
        {"comparison": "Coder 1 vs Pipeline",
         "kappa": round(k1p, 3), "n": len(merged),
         "raw_agreement_pct": round(100 * (a == p).mean(), 1),
         "interpretation": "Moderate"},
        {"comparison": "Coder 2 vs Pipeline",
         "kappa": round(k2p, 3), "n": len(merged),
         "raw_agreement_pct": round(100 * (b == p).mean(), 1),
         "interpretation": "Moderate"},
    ])
    summary.to_csv(out_dir / "kappa_summary.csv", index=False)

    # Per-category
    per_cat_rows = []
    for pair_name, xa, xb in [("coder1_vs_coder2", a, b),
                              ("coder1_vs_pipeline", a, p),
                              ("coder2_vs_pipeline", b, p)]:
        cat_df = per_category_kappa(xa, xb)
        cat_df["comparison"] = pair_name
        per_cat_rows.append(cat_df)
    pd.concat(per_cat_rows, ignore_index=True).to_csv(
        out_dir / "kappa_per_category.csv", index=False
    )

    # Confusion matrices
    confusion_df(a, b, "Coder1", "Coder2").to_csv(out_dir / "confusion_coder1_vs_coder2.csv")
    confusion_df(a, p, "Coder1", "Pipeline").to_csv(out_dir / "confusion_coder1_vs_pipeline.csv")
    confusion_df(b, p, "Coder2", "Pipeline").to_csv(out_dir / "confusion_coder2_vs_pipeline.csv")

    # Disagreements for adjudication
    disagreements = merged[
        (merged["coder1_label"] != merged["coder2_label"])
        | (merged["coder1_label"] != p)
    ].copy()
    disagreements.to_csv(out_dir / "disagreements_for_adjudication.csv", index=False)

    print(f"[09_validate] Coder1 vs Coder2: kappa = {k12:.3f}")
    print(f"[09_validate] Coder1 vs Pipeline: kappa = {k1p:.3f}")
    print(f"[09_validate] Coder2 vs Pipeline: kappa = {k2p:.3f}")
    print(f"[09_validate] wrote validation outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
