"""
06_pca_and_field_test.py
------------------------
Compute PCA on episode profiles and run the six pre-specified metrics for the
field-versus-typology hypothesis test (Table 8).

Outputs (in --output-dir):
    - pca_loadings.csv                (Table 6)
    - pca_scree.csv                   (variance explained)
    - pca_axis_interpretation.csv     (Table 7)
    - pca_episode_scores.csv          (per-episode PC1..PC7 scores)
    - field_vs_typology_test.csv      (Table 8)

Usage:
    python scripts/06_pca_and_field_test.py \\
        --profiles episode_level/episode_profiles.csv \\
        --regimes paper_evidence/kmeans_regimes_k5.csv \\
        --louvain paper_evidence/louvain_communities.csv \\
        --output-dir paper_evidence/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

from utils import js_similarity


CATEGORIES = [
    "financial", "market", "institutional", "capability",
    "talent", "network", "self_imposed",
]


AXIS_INTERPRETATIONS = {
    "PC1": {
        "positive_pole": "Market, Financial",
        "negative_pole": "Talent, Capability",
        "interpretation": "External resources vs. internal capacity",
    },
    "PC2": {
        "positive_pole": "Financial",
        "negative_pole": "Market",
        "interpretation": "Supply-side vs. demand-side resources",
    },
    "PC3": {
        "positive_pole": "Capability",
        "negative_pole": "Talent",
        "interpretation": "Knowledge vs. people",
    },
}


def profile_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [f"prop_constraint_{c}" for c in CATEGORIES]
    return df[cols].to_numpy()


def dominant_assignment(X: np.ndarray) -> np.ndarray:
    return np.argmax(X, axis=1)


def modularity(labels: np.ndarray, S: np.ndarray, threshold: float = 0.55) -> float:
    G = nx.Graph()
    for i in range(len(S)):
        G.add_node(i)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if S[i, j] >= threshold:
                G.add_edge(i, j, weight=float(S[i, j]))
    communities = {}
    for i, c in enumerate(labels):
        communities.setdefault(int(c), set()).add(i)
    return float(nx.algorithms.community.quality.modularity(G, communities.values()))


def js_similarity_matrix(X: np.ndarray) -> np.ndarray:
    n = len(X)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = js_similarity(X[i], X[j])
            S[i, j] = s
            S[j, i] = s
        S[i, i] = 1.0
    return S


def verdict(observed: float, typology_range: tuple[float, float | None],
            field_range: tuple[float, float]) -> str:
    lo_t, hi_t = typology_range
    lo_f, hi_f = field_range
    in_typology = (hi_t is None and observed > lo_t) or (hi_t is not None and lo_t <= observed <= hi_t)
    in_field = lo_f <= observed <= hi_f
    if in_typology and not in_field:
        return "TYPOLOGY"
    if in_field and not in_typology:
        return "FIELD"
    if in_typology and in_field:
        return "AMBIGUOUS"
    return "NEITHER"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--regimes", required=True, help="kmeans_regimes_k5.csv")
    ap.add_argument("--louvain", required=True, help="louvain_communities.csv")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.profiles)
    X = profile_matrix(df)
    regimes = pd.read_csv(args.regimes)
    louv = pd.read_csv(args.louvain)

    kmeans_labels = regimes["cluster"].to_numpy()
    louv_labels = louv["louvain_community"].to_numpy()
    dom_labels = dominant_assignment(X)

    # --- PCA ---
    pca = PCA(n_components=7)
    scores = pca.fit_transform(X)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=CATEGORIES,
        columns=[f"PC{i+1}" for i in range(7)],
    ).round(3)
    loadings.to_csv(out_dir / "pca_loadings.csv")

    scree = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(7)],
        "variance_explained": pca.explained_variance_ratio_.round(3),
        "cumulative": pca.explained_variance_ratio_.cumsum().round(3),
    })
    scree.to_csv(out_dir / "pca_scree.csv", index=False)

    axis_interp = pd.DataFrame([
        {
            "PC": k,
            "variance": f"{scree.iloc[i]['variance_explained']*100:.1f}%",
            **AXIS_INTERPRETATIONS[k],
        }
        for i, k in enumerate(["PC1", "PC2", "PC3"])
    ])
    axis_interp.to_csv(out_dir / "pca_axis_interpretation.csv", index=False)

    scores_df = df[["episode_id"]].copy()
    for i in range(7):
        scores_df[f"PC{i+1}"] = scores[:, i]
    scores_df.to_csv(out_dir / "pca_episode_scores.csv", index=False)
    print(f"[06_pca] wrote pca_loadings.csv, pca_scree.csv, pca_axis_interpretation.csv")

    # --- Field vs Typology test ---
    S = js_similarity_matrix(X)

    sil = float(silhouette_score(X, kmeans_labels))
    mod_km = modularity(kmeans_labels, S, 0.55)
    mod_lv = modularity(louv_labels, S, 0.55)
    ari_km_lv = float(adjusted_rand_score(kmeans_labels, louv_labels))
    ari_km_dom = float(adjusted_rand_score(kmeans_labels, dom_labels))
    ari_lv_dom = float(adjusted_rand_score(louv_labels, dom_labels))

    rows = [
        {"metric": "Silhouette (k=5)",
         "typology_range": "> 0.50", "field_range": "0.15-0.45",
         "observed": round(sil, 2),
         "verdict": verdict(sil, (0.50, None), (0.15, 0.45))},
        {"metric": "Modularity (KMeans)",
         "typology_range": "> 0.40", "field_range": "0.15-0.40",
         "observed": round(mod_km, 2),
         "verdict": verdict(mod_km, (0.40, None), (0.15, 0.40))},
        {"metric": "Modularity (Louvain)",
         "typology_range": "> 0.40", "field_range": "0.15-0.40",
         "observed": round(mod_lv, 2),
         "verdict": verdict(mod_lv, (0.40, None), (0.15, 0.40))},
        {"metric": "ARI: KMeans vs Louvain",
         "typology_range": "> 0.70", "field_range": "0.20-0.55",
         "observed": round(ari_km_lv, 2),
         "verdict": verdict(ari_km_lv, (0.70, None), (0.20, 0.55))},
        {"metric": "ARI: KMeans vs dominant",
         "typology_range": "> 0.70", "field_range": "0.20-0.55",
         "observed": round(ari_km_dom, 2),
         "verdict": verdict(ari_km_dom, (0.70, None), (0.20, 0.55))},
        {"metric": "ARI: Louvain vs dominant",
         "typology_range": "> 0.70", "field_range": "0.20-0.55",
         "observed": round(ari_lv_dom, 2),
         "verdict": verdict(ari_lv_dom, (0.70, None), (0.20, 0.55))},
    ]
    pd.DataFrame(rows).to_csv(out_dir / "field_vs_typology_test.csv", index=False)
    print(f"[06_pca] wrote field_vs_typology_test.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
