# Constraint as Structural Field

**Replication materials for**: *Constraint as Structural Field: A Computational Analysis of Entrepreneurial Discourse*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-full-brightgreen.svg)](#reproducibility)

---

## Overview

This repository contains the complete computational pipeline, data schemas, validation artifacts, and analysis outputs for the SCEN-EC (Structural Constraint-Embedding Network analysis of Entrepreneurial Cognition) project. The paper argues that entrepreneurial constraint is better represented as position in a continuous structural field than as membership in discrete typological categories, and demonstrates this empirically through a computational analysis of 125 entrepreneurial podcast episodes.

**Key findings**:
- Five inductively derived constraint regimes (financial, market, capability, talent, institutional) stable across k = 4 to 6
- Three interpretable compositional axes accounting for 70.4% of variance
- Pre-specified hypothesis test yields 4-of-6 field verdicts, 0 typology verdicts
- Eight major industries occupy distinctive positional signatures

---

## Repository Structure

```
Constraint-as-Structural-Field/
├── README.md                              # This file
├── LICENSE                                # MIT license
├── CITATION.cff                           # Citation metadata
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
│
├── notebooks/
│   └── SCEN-EC-02-Github.ipynb            # Main pipeline (Colab-ready)
│
├── scripts/
│   ├── 01_extract_transcripts.py          # Segment .docx transcripts into sentences
│   ├── 02_build_embeddings.py             # Generate sentence embeddings
│   ├── 03_classify_constraints.py         # Three-stage classification pipeline
│   ├── 04_episode_profiles.py             # Aggregate to episode-level 7-dim vectors
│   ├── 05_cluster_regimes.py              # KMeans + Louvain + k-selection
│   ├── 06_pca_and_field_test.py           # PCA + field-vs-typology hypothesis test
│   ├── 07_industry_analysis.py            # Lift analysis for industry signatures
│   ├── 08_robustness_analysis.py          # Multi-axis sensitivity analysis
│   ├── 09_validate_kappa.py               # Cohen's kappa on validation sample
│   └── utils.py                           # Shared helpers
│
├── data/
│   ├── constraint_codebook.json           # Seven-category codebook (Table 1)
│   ├── episode_metadata.csv               # Episode-to-guest-to-industry mapping
│   └── README.md                          # Data documentation
│
├── validation/
│   ├── CODING_INSTRUCTIONS.txt            # Codebook handout for human coders
│   ├── validation_sample.csv              # 230-sentence stratified sample (with pipeline labels)
│   ├── validation_sample_CODER1.csv       # Blank coder 1 file
│   ├── validation_sample_CODER2.csv       # Blank coder 2 file
│   ├── validation_sample_CODER1_completed.csv  # Coder 1 completed labels
│   ├── validation_sample_CODER2_completed.csv  # Coder 2 completed labels
│   ├── kappa_summary.csv                  # Table 2 kappa values
│   ├── kappa_per_category.csv             # Per-category kappa (Appendix C)
│   ├── confusion_coder1_vs_coder2.csv     # Confusion matrix (Appendix D)
│   ├── confusion_coder1_vs_pipeline.csv   # Confusion matrix
│   ├── confusion_coder2_vs_pipeline.csv   # Confusion matrix
│   └── disagreements_for_adjudication.csv # Cases requiring adjudication
│
├── episode_level/
│   ├── episode_profiles.csv               # 111 episodes × 7 constraint proportions
│   └── episode_industry_labels.csv        # Episode-to-industry assignments
│
├── paper_evidence/
│   ├── field_vs_typology_test.csv         # Table 8: pre-specified hypothesis test
│   ├── regime_generalization_k4_5_6.csv   # Table 5: regime stability across k
│   ├── pca_loadings.csv                   # Table 6: PCA loadings
│   ├── pca_axis_interpretation.csv        # Table 7: axis interpretations
│   ├── pca_scree.csv                      # Variance explained per PC
│   └── bridge_entrepreneurs.csv           # Table 9: bridge episodes
│
├── constraint_x_industry/
│   ├── Q3_lift_industry_x_constraint.csv  # Full lift matrix
│   └── Q3_industry_signatures.csv         # Table 10: industry signatures
│
├── robustness/
│   ├── k_selection_metrics.csv            # Table 4: silhouette/CH/DB/bootstrap
│   ├── sensitivity_analysis.csv           # Full per-run sensitivity data
│   └── sensitivity_summary_by_axis.csv    # Table 11: sensitivity summary
│
├── figures/
│   ├── fig_kmeans_centroids.png           # Figure 1
│   ├── fig_k_selection.png                # Figure 2
│   ├── fig_per_cluster_silhouette.png     # Figure 3
│   ├── fig_pca_scree_loadings.png         # Figure 4
│   ├── fig_episode_pca.png                # Figure 5
│   ├── fig_episode_network.png            # Figure 6
│   ├── fig_Q2_industry_constraint_profile.png  # Figure 7
│   ├── fig_Q3_lift_industry_constraint.png     # Figure 8
│   ├── fig_sensitivity_analysis.png       # Figure 9
│   └── fig_episode_profile_heatmap.png    # Supplementary
│
└── outputs/
    └── constraint_reclassification_v2.csv # Sentence-level final labels
```

---

## Reproducibility

### Environment

- **Python**: 3.10 or higher
- **Runtime**: ~90 minutes on a standard Colab CPU instance (no GPU required)
- **Memory**: 4 GB minimum

### Quick Start (Google Colab)

The fastest path to full reproduction is to open the notebook in Colab:

1. Open [`notebooks/SCEN-EC-02-Github.ipynb`](notebooks/SCEN-EC-02-Github.ipynb) in Google Colab
2. Mount Google Drive when prompted (transcripts must be placed at the expected path)
3. Run all cells top to bottom

### Local Installation

```bash
# Clone the repository
git clone https://github.com/GelarehGF/Constraint-as-Structural-Field.git
cd Constraint-as-Structural-Field

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline Locally

The pipeline is organized as nine sequential scripts in `scripts/`. Each script reads its input from the outputs of the prior step and writes its output to the canonical folder:

```bash
# Step 1: Extract sentences from raw transcripts
python scripts/01_extract_transcripts.py --input path/to/transcripts/ --output outputs/sentences.csv

# Step 2: Generate embeddings
python scripts/02_build_embeddings.py --input outputs/sentences.csv --output outputs/embeddings.npy

# Step 3: Classify constraint sentences
python scripts/03_classify_constraints.py --sentences outputs/sentences.csv --embeddings outputs/embeddings.npy --codebook data/constraint_codebook.json --output outputs/constraint_reclassification_v2.csv

# Step 4: Aggregate to episode profiles
python scripts/04_episode_profiles.py --input outputs/constraint_reclassification_v2.csv --output episode_level/episode_profiles.csv

# Step 5: Cluster into regimes
python scripts/05_cluster_regimes.py --profiles episode_level/episode_profiles.csv --output-dir paper_evidence/

# Step 6: PCA and field-vs-typology test
python scripts/06_pca_and_field_test.py --profiles episode_level/episode_profiles.csv --output-dir paper_evidence/

# Step 7: Industry analysis
python scripts/07_industry_analysis.py --profiles episode_level/episode_profiles.csv --industries episode_level/episode_industry_labels.csv --output-dir constraint_x_industry/

# Step 8: Robustness analysis
python scripts/08_robustness_analysis.py --profiles episode_level/episode_profiles.csv --output-dir robustness/

# Step 9: Validate against human coders
python scripts/09_validate_kappa.py --coder1 validation/validation_sample_CODER1_completed.csv --coder2 validation/validation_sample_CODER2_completed.csv --master validation/validation_sample.csv --output-dir validation/
```

### Transcripts

The 125 podcast transcripts are **not redistributed in this repository** because the audio is hosted by the Innovation Fuel program. Episodes are publicly accessible at the program's hosted location. File names in the pipeline match the public episode identifiers exactly, so any user can independently reconstruct the corpus by downloading the transcripts from the program site.

To run the full pipeline from scratch, place the transcript `.docx` files in a folder named `transcripts/` at the repository root. The classification pipeline is designed to be corpus-agnostic: pointing it at any comparable narrative corpus with the same codebook will produce structurally comparable outputs.

---

## What Each Output File Contains

### Table 2 (Validation): `validation/kappa_summary.csv`

| Comparison | Kappa | n | Raw agreement | Interpretation |
|---|---|---|---|---|
| Coder 1 vs. Coder 2 | 0.838 | 230 | 86.5% | Almost perfect |
| Coder 1 vs. Pipeline | 0.491 | 230 | 55.2% | Moderate |
| Coder 2 vs. Pipeline | 0.446 | 230 | 50.9% | Moderate |

Inter-coder agreement was computed between two coding passes with deliberately differentiated priors (strict, permissive). We treat 0.838 as a measure of codebook discriminability rather than as an independent inter-rater statistic. Per-category kappa values in `kappa_per_category.csv` show that explicit-signal categories (financial, talent, institutional, market) achieve kappa above 0.60 against the pipeline, while diffuse-signal categories (self-imposed, network) show weaker agreement.

### Table 5 (k-Generalization): `paper_evidence/regime_generalization_k4_5_6.csv`

Confirms that the five primary regimes are not k-dependent:
- k = 4: financial, market, capability, talent (institutional absorbed)
- k = 5: financial, market, capability, talent, institutional
- k = 6: financial, market, capability, talent, institutional + self-imposed emerges

### Table 8 (Field vs. Typology): `paper_evidence/field_vs_typology_test.csv`

Six pre-specified discriminating metrics with observed values and verdicts (FIELD / TYPOLOGY / NEITHER). Result: 4/6 FIELD, 1 TYPOLOGY (geometric tautology), 1 NEITHER.

### Table 10 (Industry Signatures): `constraint_x_industry/Q3_industry_signatures.csv`

Constraint lift by industry, log₂ scale. Reveals distinctive positional signatures for each of eight represented industries.

### Table 11 (Robustness): `robustness/sensitivity_summary_by_axis.csv`

Multi-axis sensitivity analysis. Three of four parameter axes fully stable; the fourth (similarity threshold) preserves regime dominants but shifts episode-level assignments as boundaries tighten — the expected signature of a field with fuzzy rather than sharp boundaries.

---

## Data Documentation

### Constraint Codebook (`data/constraint_codebook.json`)

Seven categories with definitions, keyword lists, multiword phrases, and exclusion rules:

- **Financial**: capital, funding, cost, affordability, credit
- **Market**: customer demand, competition, market size, adoption
- **Institutional**: regulation, law, policy, compliance, licensing
- **Capability**: internal knowledge, expertise, learning, technical skill
- **Talent**: hiring, recruiting, retaining, finding qualified people
- **Network**: industry connections, partnerships, mentors, ecosystem access
- **Self-imposed**: voluntary limitations from values, ethics, mission

Three exclusion rules govern boundaries:
1. "couldn't afford to hire X" → financial (not talent)
2. "didn't know how to do X" → capability (not talent)
3. "chose not to for ethical reasons" → self-imposed (regardless of surface category)

### Episode Profiles (`episode_level/episode_profiles.csv`)

Each row is one of the 111 analyzable episodes; columns are the proportion of constraint-bearing sentences assigned to each category. Columns:

| Column | Description |
|---|---|
| `episode_id` | Public episode identifier |
| `n_constraint_sentences` | Number of constraint-bearing sentences |
| `prop_constraint_financial` | Proportion classified as financial |
| `prop_constraint_market` | Proportion classified as market |
| `prop_constraint_institutional` | Proportion classified as institutional |
| `prop_constraint_capability` | Proportion classified as capability |
| `prop_constraint_talent` | Proportion classified as talent |
| `prop_constraint_network` | Proportion classified as network |
| `prop_constraint_self_imposed` | Proportion classified as self-imposed |

---

## Validation Workflow

The validation sample was constructed by stratified random selection: approximately 30 sentences per primary category plus 20 unclassified sentences (total n = 230). To reproduce or extend the validation:

1. Open `validation/CODING_INSTRUCTIONS.txt` and review the codebook
2. Have two independent coders complete `validation_sample_CODER1.csv` and `validation_sample_CODER2.csv` without consulting each other or the pipeline labels
3. Save the completed files as `*_completed.csv` in the same folder
4. Run `python scripts/09_validate_kappa.py` to compute Cohen's kappa and generate confusion matrices

---

## Generative AI Disclosure

It was not used for data analysis, coding, statistical computation, or interpretation of results. The constraint codebook and all analytical decisions were developed by the authors. All content, methods, and conclusions are the sole responsibility of the authors.

The sentence-transformers/all-MiniLM-L6-v2 model is used within the pipeline as a deterministic feature extractor for sentence embeddings — a generative AI component only in the technical sense that it is a transformer-based model. It is not used to generate any text presented in the paper.

---

## Citation

If you use this pipeline or data in your research, please cite:

```bibtex
@article{ConstraintField2026,
  title   = {Constraint as Structural Field: A Computational Analysis of Entrepreneurial Discourse},
  author  = {[Author Name]},
  journal = {[Journal Name]},
  year    = {2026},
  url     = {https://github.com/GelarehGF/Constraint-as-Structural-Field}
}
```

See `CITATION.cff` for machine-readable citation metadata.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

For questions about the pipeline, please open an issue on this repository.
For questions about the paper, contact [author email].
