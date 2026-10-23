# Protein Localization: ETL and Classification

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-ETL-150458?logo=pandas&logoColor=white)
![PyArrow](https://img.shields.io/badge/PyArrow-Parquet-4B8BBE)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU-EC6C00)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Data Source](https://img.shields.io/badge/Data-UniProt-1F6FEB)
![Data License](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey)
![Kaggle](https://img.shields.io/badge/Kaggle-Datasets-20BEFF?logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

![](https://github.com/RafaelGallo/bioinfo_protein_ETL/blob/main/img/log1.jpg?raw=true)

An end-to-end ETL pipeline that builds machine-learning-ready datasets from the
[UniProt](https://www.uniprot.org) protein knowledgebase, plus baseline models that
predict whether a protein is located in the **cell nucleus** from its amino acid
sequence alone.

The project produces two datasets, both intended for publication on Kaggle:

1. **Classification dataset:** reviewed (Swiss-Prot) proteins with a binary target
   (`1` = nuclear, `0` = not nuclear).
2. **New proteins dataset:** unreviewed (TrEMBL) proteins with no subcellular location
   annotation, used as an unlabeled set where a trained model predicts new proteins.

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Target Definition](#target-definition)
- [Data Quality Steps](#data-quality-steps)
- [Modeling](#modeling)
- [Results](#results)
- [Known Limitations](#known-limitations)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [SQL Database](#sql-database)
- [Publishing to Kaggle](#publishing-to-kaggle)
- [Roadmap](#roadmap)
- [Data License and Citation](#data-license-and-citation)
- [Author](#author)

## Overview

| Item | Description |
|---|---|
| Goal | Predict nuclear localization of proteins from sequence |
| Data source | UniProt REST API (Swiss-Prot and TrEMBL) |
| Task | Binary classification (one-vs-rest) |
| Input | Amino acid sequence and derived features |
| Storage | Parquet, CSV and SQLite |
| Models | Logistic Regression and XGBoost (GPU) |

## Pipeline

```
UniProt REST API
      |
      v
 extract.py / extract_new.py       raw CSV (input/)
      |
      v
 csv_to_parquet.py                 raw Parquet (data/)
      |
      v
 create_target.py                  clean dataset + target
      |                            Parquet + CSV (data/) + SQLite (sql/)
      v
 model.py / model_xgb.py           metrics, predictions, saved models (output/)
```

`main.py` orchestrates the ETL steps. The models are run separately and are not part of
the ETL pipeline.

## Project Structure

```
uniprot/
    main.py                  ETL orchestrator
    py/                      pipeline scripts
        extract.py           labeled proteins (Swiss-Prot) via cursor pagination
        extract_new.py       unannotated proteins (TrEMBL)
        csv_to_parquet.py    chunked CSV to Parquet conversion
        create_target.py     target creation, cleaning, Parquet/CSV/SQLite load
        model.py             Logistic Regression baseline
        model_xgb.py         XGBoost (GPU) with predictions saved to CSV and SQL
        publish_kaggle.py    builds Kaggle dataset folders and metadata
    input/                   raw CSV files from the API
    data/                    Parquet and final CSV files
    sql/
        uniprot.db           SQLite database
    output/                  models, metrics and predictions
    kaggle/                  dataset folders ready for upload
    requirements.txt
```

## Dataset

Raw extraction fields (`reviewed:true`, sequence length between 50 and 1500):

| Column | Description |
|---|---|
| `Entry` | UniProt accession |
| `Entry Name` | UniProt entry name |
| `Protein names` | Recommended and alternative protein names |
| `Gene Names` | Associated gene names |
| `Organism` | Organism name |
| `Organism (ID)` | NCBI taxonomy ID |
| `Length` | Sequence length in amino acids |
| `Sequence` | Amino acid sequence |
| `Subcellular location [CC]` | Annotated subcellular location text |
| `EC number` | Enzyme Commission number |
| `Gene Ontology IDs` | GO terms (removed from the final dataset) |
| `Pfam` | Pfam family cross-references |

Dataset sizes:

| Stage | Rows |
|---|---|
| Raw extraction (Swiss-Prot) | 348,465 |
| After cleaning | 301,309 |
| New proteins (TrEMBL, unlabeled) | 100,000 |

## Target Definition

The `target` column is created from the UniProt subcellular location text:

- `1` if any annotated location matches `\bnucle(?:us|ar)\b` (case insensitive).
- `0` otherwise.

Evidence tags (`{ECO:...}`), the `SUBCELLULAR LOCATION:` prefix and `Note=` sections are
removed before matching, so only the annotated locations are used. A proteins located in
several compartments (for example nucleus and cytoplasm) is labeled `1`.

Configurable options in `create_target.py`:

| Option | Default | Effect |
|---|---|---|
| `target_name` / `target_pattern` | nucleus | Change them to build another one-vs-rest target |
| `strict_nucleus` | `False` | Excludes proteins located only in the nuclear envelope or membrane |
| `only_eukaryote_proxy` | `False` | Keeps only organisms with at least one positive protein |

## Data Quality Steps

1. Drop rows without location text or sequence.
2. Keep only sequences made of the 20 standard amino acids (removes 1,534 sequences).
3. Remove sequences with conflicting labels and exact duplicate sequences (removes 45,622 rows).
4. Drop `Gene Ontology IDs`, because the GO term `GO:0005634` (nucleus) leaks the target.
5. Report class balance (14.30% positive after cleaning).

## Modeling

Models use **only sequence-derived features** (28 in total). Identifiers, protein names,
the location text and GO terms are never used as features.

| Feature group | Features |
|---|---|
| Amino acid composition | 20 frequencies (`aa_A` ... `aa_Y`) |
| Physicochemical | `log_length`, `gravy` (Kyte-Doolittle), `frac_positive`, `frac_negative`, `net_charge_frac`, `frac_aromatic` |
| Nuclear localization signal motifs | `kr_motif_count` (4 or more consecutive K/R), `kr_motif_pat7` (classic monopartite pattern) |

Evaluation design:

- **Group split by organism** (`GroupShuffleSplit`): 60% train, 20% validation, 20% test,
  with no organism shared between sets.
- **Class imbalance:** `class_weight="balanced"` (Logistic Regression) and
  `scale_pos_weight` (XGBoost).
- **Threshold tuning on validation only:** the classification threshold maximizes F1 on the
  validation set and is then applied once to the test set.
- **Early stopping** on validation PR-AUC (XGBoost).
- Main metrics are PR-AUC and F1, not accuracy.

## Known Limitations

- **Homology leakage:** splitting by organism does not separate orthologs (for example
  the same protein in human and mouse), so reported metrics are likely optimistic. A split by
  sequence similarity (MMseqs2 or CD-HIT at 30% identity) is planned.
- **Organism effect:** 53.9% of the cleaned rows come from organisms with no nuclear protein
  (mainly bacteria and archaea). The eukaryote proxy filter removes them, but it uses the
  target itself. Filtering by taxonomic lineage is a better solution.
- **Label definition:** the regex also marks nuclear envelope and membrane annotations as
  nuclear. Use `strict_nucleus = True` for a stricter definition.
- **Sequence-only features:** composition and motif features are a baseline. Protein language
  model embeddings (ESM-2, ProtBERT) are expected to perform better.

## Getting Started

Requirements: Python 3.11 or newer. A CUDA-capable NVIDIA GPU is optional (used by XGBoost).

```bash
git clone https://github.com/YOUR_USER/uniprot-protein-localization.git
cd uniprot-protein-localization

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install requests pandas pyarrow tqdm scikit-learn xgboost joblib kaggle
```

## Usage

Run the full ETL:

```bash
python main.py
```

Useful options:

```bash
python main.py --list                              # show steps and script status
python main.py --skip extract extract_new          # skip the slow API extraction
python main.py --steps create_target               # run a single step
python main.py --steps publish                     # prepare the Kaggle folders
```

Run the models separately:

```bash
python py/model.py          # Logistic Regression
python py/model_xgb.py      # XGBoost on GPU
```

Outputs:

| File | Description |
|---|---|
| `data/uniprot_target.parquet` and `.csv` | Final classification dataset |
| `sql/uniprot.db` | SQLite database with dataset and predictions |
| `output/model_xgb.joblib` | Trained model, threshold and feature names |
| `output/metrics_xgb.json` | Test metrics |
| `output/predictions_xgb_all.csv` | 0/1 prediction for every protein, with `split` column |
| `output/predictions_xgb_test.csv` | Predictions for the test set only |

Use only rows with `split == "test"` to evaluate the model. Predictions on `train` rows are
optimistic.

## SQL Database

`sql/uniprot.db` is a SQLite file with two tables:

| Table | Content |
|---|---|
| `uniprot_target` | Clean dataset with snake_case column names (`entry`, `organism_id`, `sequence`, `target`, ...) |
| `predictions_xgb` | `entry`, `organism_id`, `length`, `target`, `proba`, `pred`, `correct`, `split` |

Example queries:

```sql
-- accuracy by split
SELECT split, COUNT(*) AS proteins, ROUND(AVG(correct), 3) AS accuracy
FROM predictions_xgb
GROUP BY split;

-- top test proteins predicted as nuclear
SELECT p.entry, u.protein_names, u.organism, p.proba
FROM predictions_xgb p
JOIN uniprot_target u ON u.entry = p.entry
WHERE p.split = 'test' AND p.pred = 1
ORDER BY p.proba DESC
LIMIT 20;
```

Datasets are created as private by default, so you can review them before making them public.

## Roadmap

- [ ] Compare Logistic Regression and XGBoost on the same split
- [ ] Homology-aware split with MMseqs2 (30% identity)
- [ ] Filter eukaryotes by taxonomic lineage instead of the proxy
- [ ] Protein language model embeddings (ESM-2) as features
- [ ] `predict.py` for the new proteins dataset
- [ ] Multi-class localization (nucleus, cytoplasm, membrane, mitochondrion, secreted)
- [ ] Publish both datasets and a demo notebook on Kaggle

## Data License and Citation

Protein data comes from [UniProt](https://www.uniprot.org) and is distributed under the
[Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license. If you use the datasets, please cite:

> The UniProt Consortium. UniProt: the Universal Protein Knowledgebase. *Nucleic Acids Research*.

The code in this repository is released under the license of your choice (for example MIT).
Add a `LICENSE` file to the repository root.

## Author

**Rafael**, data scientist based in Sao Paulo, Brazil.

- GitHub: [@Rafael Gallo](https://github.com/RafaelGallo)
- Kaggle: [Rafael Gallo](https://www.kaggle.com/gallo33henrique)
- LinkedIn: [Rafael Gallo](https://www.linkedin.com/in/rafael-g-986a73150/)
