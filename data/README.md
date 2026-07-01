# Data

This folder contains the codebook and episode-level metadata needed to run the pipeline. Raw transcripts are **not** included; see the Transcripts section of the main README for how to obtain them.

## Files

### `constraint_codebook.json`
The seven-category codebook operationalizing constraints. Contains for each category:
- `definition`: one-sentence conceptual definition
- `keywords`: single-word signals used by the keyword-boost step in Stage 3 classification
- `multiword_phrases`: exact multi-word patterns treated as strong signals
- `prototype_sentences`: canonical exemplars used to compute cosine similarity in Stage 3
- `representative_phrases`: illustrative phrases reported in Table 1 of the paper

Also contains:
- `similarity_threshold`: 0.42 (Stage 3 classification threshold)
- `gate_threshold`: 0.40 (Stage 2 constraint-bearing gate threshold)
- `exclusion_rules`: three deterministic post-classification correction rules
- `meta_filter`: parameters for Stage 1 (min length, boilerplate patterns)
- `gate_prototypes_for_embedding_fallback`: 15 canonical constraint statements used by the Stage 2 embedding-based gate

### `episode_metadata.csv`
One row per episode with columns:
- `episode_id`: public episode identifier (matches transcript filename)
- `guest_name`: named guest(s)
- `guest_company`: guest's company affiliation
- `industry`: assigned industry (technology, finance, retail, healthcare, education, agriculture, manufacturing, services, energy, or unknown)
- `n_sentences_raw`: total sentences after segmentation
- `n_sentences_gated`: sentences passing the Stage 2 gate
- `n_constraint_sentences`: sentences classified into one of the seven categories

## Provenance

The codebook was developed iteratively: an initial version drawn from resource-based, institutional, and bricolage literatures; refined through pilot coding of 20 episodes; and finalized through inter-coder disagreement adjudication on the 230-sentence validation sample. See Section 3.2.1 of the paper for the full development process.

## Reproducing episode_metadata.csv

The industry assignment in `episode_metadata.csv` is produced by `scripts/07_industry_analysis.py` from:
1. The program's official guest list (mapping episode → guest → company)
2. Company-to-industry classifications (from company websites; when ambiguous, transcript content is used as a tiebreaker)

Users reconstructing the corpus from scratch should generate their own `episode_metadata.csv` following this two-step process.
