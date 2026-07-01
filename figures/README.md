# Figures

This folder contains the nine figures referenced in the paper. All figures are
regenerable from the CSV outputs in the sibling folders using standard matplotlib
plotting (code in the main notebook, Section 4).

## Expected files

| Filename | Paper reference | Source data |
|---|---|---|
| `fig_kmeans_centroids.png` | Figure 1 | `paper_evidence/kmeans_regimes_k5_summary.csv` |
| `fig_k_selection.png` | Figure 2 | `robustness/k_selection_metrics.csv` |
| `fig_per_cluster_silhouette.png` | Figure 3 | `paper_evidence/kmeans_regimes_k5.csv` + episode profiles |
| `fig_pca_scree_loadings.png` | Figure 4 | `paper_evidence/pca_scree.csv`, `pca_loadings.csv` |
| `fig_episode_pca.png` | Figure 5 | `paper_evidence/pca_episode_scores.csv` |
| `fig_episode_network.png` | Figure 6 | Jensen-Shannon similarity matrix (regenerable) |
| `fig_Q2_industry_constraint_profile.png` | Figure 7 | `constraint_x_industry/Q2_industry_constraint_profile.csv` |
| `fig_Q3_lift_industry_constraint.png` | Figure 8 | `constraint_x_industry/Q3_lift_industry_x_constraint.csv` |
| `fig_sensitivity_analysis.png` | Figure 9 | `robustness/sensitivity_analysis.csv` |

## Optional supplementary
| Filename | Purpose |
|---|---|
| `fig_episode_profile_heatmap.png` | Full episode profile heatmap (Appendix figure) |

## Reproducing the figures

Run the notebook cells that follow the analytical sections. Each figure is generated
using deterministic matplotlib code with seaborn styling. For rendering consistency,
use `matplotlib >= 3.7.0` and `seaborn >= 0.12.0` (see `requirements.txt`).
