"""
05_cluster_regimes.py
---------------------
KMeans clustering with k-selection diagnostics.
Also produces the Louvain community detection on the Jensen-Shannon similarity network.

Outputs (in --output-dir):
    - kmeans_regimes_k5.csv          (episode_id, cluster, dominant, mean_p_dominant)
    - k_selection_metrics.csv         (Table 4)
    - regime_generalization_k4_5_6.csv  (Table 5)
    - louvain_communities.csv         (episode_id, louvain_community)

Usage:
    python scripts/05_cluster_regimes.py \\
        --profiles episode_level/episode_profiles.csv \\
        --output-dir paper_evidence/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score,
)

from utils import js_similarity


CATEGORIES = [
    "financial", "market", "institutional", "capability",
    "talent", "network", "self_imposed",
]


def profile_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [f"prop_constraint_{c}" for c in CATEGORIES]
    return df[cols].to_numpy()


def bootstrap_ari(X: np.ndarray, k: int, n_boot: int = 10, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ref = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).labels_
    aris = []
    for i in range(n_boot):
        idx = rng.choice(len(X), int(0.9 * len(X)), replace=False)
        boot_labels = KMeans(n_clusters=k, n_init=10, random_state=seed + i + 1).fit(X[idx]).labels_
        aris.append(adjusted_rand_score(ref[idx], boot_labels))
    return float(np.mean(aris)), float(np.std(aris))


def run_kmeans_at_k(X: np.ndarray, k: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, centroids)."""
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    return km.labels_, km.cluster_centers_


def summarize_regimes(labels: np.ndarray, centroids: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in range(len(centroids)):
        dominant_idx = int(np.argmax(centroids[k]))
        rows.append({
            "cluster": k,
            "n": int((labels == k).sum()),
            "dominant": CATEGORIES[dominant_idx],
            "mean_p_dominant": float(centroids[k][dominant_idx]),
        })
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--js-threshold", type=float, default=0.55)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.profiles)
    X = profile_matrix(df)

    # --- k-selection metrics ---
    k_metrics = []
    for k in range(3, 9):
        labels, _ = run_kmeans_at_k(X, k, args.seed)
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        m, s = bootstrap_ari(X, k, n_boot=10, seed=args.seed)
        k_metrics.append({
            "k": k,
            "silhouette": round(sil, 3),
            "calinski_harabasz": round(ch, 1),
            "davies_bouldin": round(db, 2),
            "bootstrap_ari_mean": round(m, 2),
            "bootstrap_ari_std": round(s, 2),
        })
    pd.DataFrame(k_metrics).to_csv(out_dir / "k_selection_metrics.csv", index=False)
    print(f"[05_cluster] wrote k_selection_metrics.csv")

    # --- KMeans at k = 5 (main solution) ---
    labels_5, centers_5 = run_kmeans_at_k(X, 5, args.seed)
    out_5 = df[["episode_id"]].copy()
    out_5["cluster"] = labels_5
    out_5.to_csv(out_dir / "kmeans_regimes_k5.csv", index=False)
    summarize_regimes(labels_5, centers_5).to_csv(
        out_dir / "kmeans_regimes_k5_summary.csv", index=False
    )
    print(f"[05_cluster] wrote kmeans_regimes_k5.csv (n={len(out_5)})")

    # --- Regime generalization across k = 4, 5, 6 (Table 5) ---
    gen_rows = []
    for k in (4, 5, 6):
        labels_k, centers_k = run_kmeans_at_k(X, k, args.seed)
        summ = summarize_regimes(labels_k, centers_k)
        dominants = "; ".join(
            f"{r['dominant']} (n = {r['n']})" for _, r in summ.iterrows()
        )
        gen_rows.append({"k": k, "regime_dominants": dominants})
    pd.DataFrame(gen_rows).to_csv(out_dir / "regime_generalization_k4_5_6.csv", index=False)
    print(f"[05_cluster] wrote regime_generalization_k4_5_6.csv")

    # --- Louvain on JS similarity network ---
    S = js_similarity_matrix(X)
    G = nx.Graph()
    for i in range(len(S)):
        G.add_node(i)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if S[i, j] >= args.js_threshold:
                G.add_edge(i, j, weight=S[i, j])
    partition = community_louvain.best_partition(G, random_state=args.seed)
    louv_rows = [{"episode_id": df.iloc[i]["episode_id"], "louvain_community": partition[i]}
                 for i in range(len(df))]
    pd.DataFrame(louv_rows).to_csv(out_dir / "louvain_communities.csv", index=False)
    print(f"[05_cluster] wrote louvain_communities.csv (n_communities="
          f"{len(set(partition.values()))})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
