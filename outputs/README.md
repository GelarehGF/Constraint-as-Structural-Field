# Outputs

This folder is a workspace for intermediate outputs produced by the pipeline:

- `sentences.csv` — output of `scripts/01_extract_transcripts.py`
- `embeddings.npy` — output of `scripts/02_build_embeddings.py`
- `embeddings_index.csv` — sentence-to-embedding-row mapping
- `constraint_reclassification_v2.csv` — output of `scripts/03_classify_constraints.py`

These files are large and regenerable and are ignored by git (see `.gitignore`).
Downstream scripts (04 through 09) write their outputs to their canonical folders:
`episode_level/`, `paper_evidence/`, `constraint_x_industry/`, `robustness/`,
and `validation/`.
