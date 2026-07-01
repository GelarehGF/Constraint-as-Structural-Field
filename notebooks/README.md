# Notebooks

## `SCEN-EC-02-Github.ipynb`

The main Colab notebook that runs the full end-to-end pipeline. This is the notebook
currently on the repository at the root; move it here after the reorganization.

**How to run in Colab**:
1. Open the notebook in Google Colab
2. Mount your Google Drive (the notebook expects `transcripts/` at a configured path)
3. Run all cells top to bottom (~90 minutes on standard CPU instance)

**Equivalence with scripts/**:
The notebook cells correspond one-to-one with the scripts in the `scripts/` folder:

| Notebook Section | Script |
|---|---|
| 1. Data loading & segmentation | `01_extract_transcripts.py` |
| 2. Embedding generation | `02_build_embeddings.py` |
| 3. Three-stage classification | `03_classify_constraints.py` |
| 4. Episode profile aggregation | `04_episode_profiles.py` |
| 5. K-means & Louvain clustering | `05_cluster_regimes.py` |
| 6. PCA & field-vs-typology test | `06_pca_and_field_test.py` |
| 7. Industry lift analysis | `07_industry_analysis.py` |
| 8. Multi-axis robustness | `08_robustness_analysis.py` |
| 9. Validation (Cohen's kappa) | `09_validate_kappa.py` |

Users who prefer command-line, containerized, or CI-based reproduction should run the
scripts sequentially. Users who prefer interactive exploration should use the notebook.
Both produce the same numerical results.
