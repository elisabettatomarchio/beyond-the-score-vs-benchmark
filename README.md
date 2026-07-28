## Beyond the Score: Fixed-Budget Benchmarking of Virtual Screening Integration Strategies for Decision-Centric Drug Discovery

Elisabetta Grazia Tomarchio, Rocco Buccheri, Antonio Rescifina

Code and data for benchmarking virtual screening integration strategies, submitted to the *International Journal of Molecular Sciences (IJMS)*, Special Issue *"Beyond Docking Scores: Decision-Centric Computer-Aided Drug Discovery"*.

## Overview

This repository contains the reproducible computational analysis pipeline used to benchmark twenty virtual screening (VS) integration strategies, combining three complementary methods — molecular docking, maximum common substructure (MCS) similarity, and a machine-learning QSAR (ML-QSAR) classifier — across five pharmacologically diverse protein targets (AChE, EGFR, hERG, HIV-1 protease, PPARγ).

All twenty strategies (individual methods, best-rank and worst-rank fusion, mean-rank consensus, and sequential funnels) are compared under an **identical experimental budget** at each screening fraction (1%, 5%, 10% of the library), so that recall and precision differences reflect the intrinsic effectiveness of each strategy rather than differences in candidate-list size.

## Repository contents

| File | Description |
|---|---|
| `process_pipeline.ipynb` | End-to-end pipeline for dataset curation and the ML-QSAR model: retrieval and curation of ChEMBL bioactivity data, decoy generation with LUDe, assembly of the training/test/VS databases, ECFP4 fingerprint generation, QSAR model training and calibration, performance metrics, and inference on the VS library. |
| `MCS_analysis.py` | MCS-based screening: clusters known actives (Butina clustering) into representative query scaffolds, screens the test and VS sets by MCS similarity, calibrates decision thresholds (Youden's index) on the test set, and reports ranking/classification metrics (AUC, BEDROC, EF1/5/10%, precision, recall) for both the best individual query and the max-similarity consensus. |
| `align_results.py` | Merges the docking (CNN score), MCS, and QSAR outputs per target, validates data integrity (QC checks on molecule/active counts after merging), and computes recall/precision for all 20 integration strategies at fixed experimental budget, producing the final result tables. |
| `Beyond_the_Score_VS_results_SI_IJMS.xlsx` | Supporting Information spreadsheet: per-target VS results and aggregated statistics. |
| `database` | This folder contains, for each of the five protein targets, the training, test, and virtual screening (VS) sets used by `process_pipeline.ipynb` and `MCS_analysis.py`|
| `LICENSE` | MIT License. |

## Pipeline workflow

The three scripts are meant to be run in sequence, one target at a time (AChE, EGFR, hERG, HIV-1 protease, PPARγ):

1. **`process_pipeline.ipynb`** — starting from ChEMBL bioactivity data, curates the actives, generates decoys with LUDe, builds the training/test/VS sets, computes ECFP4 fingerprints, trains and calibrates the ML-QSAR classifier, and scores the VS library, producing `VS_QSAR_predictions.csv` (QSAR probability per compound) and the labelled train/test sets used by `MCS_analysis.py`.
2. **`MCS_analysis.py`** — takes the same train/test/VS SMILES sets, clusters the training actives to obtain representative query scaffolds, calibrates thresholds on the test set, and scores the VS library by MCS similarity, producing the `consensus_max_score` per compound used downstream.
3. **`align_results.py`** — merges the docking, MCS, and QSAR scores for each target on a common set of aligned molecules, runs QC checks (expects exactly 50 confirmed actives per target after merging), and computes recall/precision for all 20 strategies at the 1%, 5%, and 10% cutoffs under matched experimental budget, producing:
   - `SINGLE_TARGET_PERFORMANCE_3M.csv` — raw per-target results. 3M stands for the three individual methods benchmarked in this work (docking, MCS similarity, and ML-QSAR).
   - `FINAL_STATISTICAL_VALIDATION_3M.csv` — mean ± SD across the five targets. 3M stands for the three individual methods benchmarked in this work (docking, MCS similarity, and ML-QSAR).
   - `QC_REPORT.csv` — merge diagnostics and data-integrity checks

> **Note:** molecular docking itself (pose generation and CNN scoring via GNINA) is performed upstream of this repository; `align_results.py` expects its output as a per-target `Enrichment_<TARGET>.csv` file (SMILES, `CNN_VS` score, experimental activity label).

## Data

The `database/` folder contains, for each of the five protein targets, the training, test, and virtual screening (VS) sets used by `process_pipeline.ipynb` and `MCS_analysis.py`:

```
database/
├── HIV1-P/
│ ├── train_metadata.csv
│ ├── test_metadata.csv 
│ └── vs_metadata.csv 
├── AChE/
│ ├── train_metadata.csv
│ ├── test_metadat.csv
│ └── vs_metadata.csv
├── EGFR/
│ ├── train_metadata.csv
│ ├── test_metadata.csv
│ └── vs_metadata.csv
├── hERG/
│ ├── train_metadata.csv
│ ├── test_metadata.csv
│ └── vs_metadata.csv
└── PPARγ/
├── train_metadata.csv
├── test_metadata.csv
└── vs_metadata.csv
```

| File | Description |
|------|-------------|
| `train_metadata.csv` | Curated training set (actives + inactive) used to train and calibrate the ML-QSAR classifier and to derive MCS query scaffolds. |
| `test_metadata.csv` | Held-out test set used for threshold calibration in MCS (Youden's index) and ML-QSAR performance evaluation (AUC, BEDROC, EF1/5/10%). |
| `vs_metadata.csv` | Virtual screening library (actives + LUDe-generated decoys) scored by docking, MCS, and QSAR, used for the fixed-budget benchmarking of the 20 integration strategies. |

## Requirements

numpy==1.26.4
pandas==2.3.3
scikit-learn==1.7.2
joblib==1.4.2



## Expected folder structure for `align_results.py`

```
project_folder/
└── training/
    ├── docking/
    │   └── Enrichment_<TARGET>.csv
    ├── graph analysis/
    │   └── <TARGET>_graph.csv
    └── <TARGET>/
        └── qsar_pipeline_results/
            └── VS_QSAR_predictions.csv
```

## Citation

If you use this code or data, please cite:

> Tomarchio, E.G., Buccheri, R., Rescifina, A. *Beyond the Score: Fixed-Budget Benchmarking of Virtual Screening Integration Strategies for Decision-Centric Drug Discovery.* Int. J. Mol. Sci., submitted.

*(DOI to be added upon publication.)*

## License

This project is released under the MIT License — see [`LICENSE`](LICENSE) for details.

## Contact

Elisabetta Grazia Tomarchio — Department of Drug and Health Sciences, University of Catania
