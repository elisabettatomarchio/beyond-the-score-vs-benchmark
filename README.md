# Beyond the Score: Fixed-Budget Benchmarking of Virtual Screening Integration Strategies for Decision-Centric Drug Discovery

Elisabetta Grazia Tomarchio, Rocco Buccheri, Antonio Rescifina

Code and data for benchmarking virtual screening integration strategies, submitted to the *International Journal of Molecular Sciences (IJMS)*, Special Issue *"Beyond Docking Scores: Decision-Centric Computer-Aided Drug Discovery"*.

## Overview

This repository contains the reproducible computational analysis pipeline used to benchmark twenty virtual screening (VS) integration strategies, combining three complementary methods — molecular docking, maximum common substructure (MCS) similarity, and a machine-learning QSAR (ML-QSAR) classifier — across five pharmacologically diverse protein targets (AChE, EGFR, hERG, HIV1-P, PPARγ).

All twenty strategies (individual methods, best-rank and worst-rank fusion, mean-rank consensus, and sequential funnels) are compared under an **identical experimental budget** at each screening fraction (1%, 5%, 10% of the library), so that recall and precision differences reflect the intrinsic effectiveness of each strategy rather than differences in candidate-list size.

The repository is self-contained: the curated training, test and virtual-screening sets are deposited together with the scripts, so every result reported in the paper can be regenerated from scratch.
## Repository contents

| File | Description |
| --- | --- |
| `process_pipeline.ipynb` | End-to-end pipeline for dataset curation and the ML-QSAR model: retrieval and curation of ChEMBL bioactivity data, decoy generation with LUDe, assembly of the training/test/VS databases, ECFP4 fingerprint generation, QSAR model training and calibration, performance metrics, and inference on the VS library. |
| `MCS_analysis.py` | MCS-based screening: clusters known actives (Butina clustering) into representative query scaffolds, screens the test and VS sets by MCS similarity, calibrates decision thresholds (Youden's index) on the test set, and reports ranking/classification metrics (AUC, BEDROC, EF1/5/10%, precision, recall) for both the best individual query and the max-similarity consensus. |
| `align_results.py` | Merges the docking (CNN score), MCS, and QSAR outputs per target, validates data integrity (QC checks on molecule/active counts after merging), computes recall/precision for all 20 integration strategies at fixed experimental budget, and runs the Friedman omnibus test across strategies, producing the final result tables. |
| `xlsx_to_inputs.py` | Convenience script for readers starting from the journal's Supporting Information rather than from a clone of this repository: regenerates the three per-target input files required by `align_results.py` directly from the spreadsheet. |
| `database` | This folder contains, for each of the five protein targets, the training, test, and virtual screening (VS) sets (SMILES and activity labels) used by `process_pipeline.ipynb` and `MCS_analysis.py`. |
| `Beyond_the_Score_VS_results_SI_IJMS.xlsx` | Spreadsheet S1 of the published Supporting Information, deposited here unchanged: per-compound VS scores, target-level fixed-budget results, aggregated statistics, ML-QSAR/MCS validation metrics, Friedman Omnibus test of strategies comparison, QC report of the merge step. |
| `requirements.txt` | Python dependencies. |
| `LICENSE` | MIT License (code). |
| `LICENSE-DATA` | CC BY-SA 3.0 (datasets derived from ChEMBL). |



## Pipeline workflow

The three scripts are meant to be run in sequence, one target at a time (AChE, EGFR, hERG, HIV1-P, PPARγ):

1. **`process_pipeline.ipynb`** — starting from ChEMBL bioactivity data, curates the actives, generates decoys with LUDe, builds the training/test/VS sets, computes ECFP4 fingerprints, trains and calibrates the ML-QSAR classifier, and scores the VS library, producing `VS_QSAR_predictions.csv` (QSAR probability per compound) and the labelled train/test sets used by `MCS_analysis.py`.
2. **`MCS_analysis.py`** — takes the same train/test/VS SMILES sets, clusters the training actives to obtain representative query scaffolds, calibrates thresholds on the test set, and scores the VS library by MCS similarity, producing the `consensus_max_score` per compound used downstream.
3. **`align_results.py`** — merges the docking, MCS, and QSAR scores for each target on a common set of aligned molecules, runs QC checks (expects exactly 50 confirmed actives per target after merging), and computes recall/precision for all 20 strategies at the 1%, 5%, and 10% cutoffs under matched experimental budget, producing:
   - `SINGLE_TARGET_PERFORMANCE_3M.csv` — raw per-target results
   - `FINAL_STATISTICAL_VALIDATION_3M.csv` — mean ± SD across the five targets
   - `FRIEDMAN_OMNIBUS_STRATEGY_COMPARISON.csv` — Friedman rank-sum test across strategies, one row per budget
   - `QC_REPORT.csv` — merge diagnostics and data-integrity checks

> **Note:** molecular docking itself (pose generation and CNN scoring via GNINA) is performed upstream of this repository; `align_results.py` expects its output as a per-target `Enrichment_<TARGET>.csv` file (SMILES, `CNN_VS` score, experimental activity label), which can be regenerated under training/docking/ via `xlsx_to_inputs.py`, or supplied by the user's own docking run.

The Friedman test is applied to the number of recovered actives, with the five benchmark targets as blocks and the twenty strategies as treatments. It is reported as a global check that the strategies are not all equivalent: no pairwise post-hoc comparison is performed, since the smallest two-sided *p* value reachable by a signed-rank test over five paired targets is 0.0625.

## Training, Test and Virtual Screening Data

The `database/` folder contains, for each of the five protein targets, the training, test, and virtual screening (VS) sets used by `process_pipeline.ipynb` and `MCS_analysis.py`:

```
database/
├── HIV1-P/
│ ├── train_metadata.csv
│ ├── test_metadata.csv 
│ └── vs_metadata.csv 
├── AChE/
│ ├── train_metadata.csv
│ ├── test_metadata.csv
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


## Reproducing the published results

To regenerate every fixed-budget result in the paper (Table 1, Figures 1–3 and the omnibus test), set `project_folder` at the bottom of `align_results.py` to the repository root and run:

```bash
pip install -r requirements.txt
python align_results.py
```

The deposited scores reproduce the published per-target table exactly, including the 300 target × cutoff × strategy records. The spreadsheet in this repository is the same file published as Spreadsheet S1 with the article, so readers who downloaded it from the journal can regenerate the same input files without cloning the repository:

```bash
python xlsx_to_inputs.py Beyond_the_Score_VS_results_SI_IJMS.xlsx ./project
```

Re-running stages 1 and 2 rebuilds the QSAR and MCS scores from the deposited training and test sets. Note that decoy generation and the scaffold-stratified splitting are stochastic, so recreating the datasets from ChEMBL rather than using the deposited ones will not reproduce them compound-for-compound.

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
## Requirements

All Python dependencies are listed in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

```
numpy==1.26.4
pandas==2.3.3
rdkit>=2024.03.1
scikit-learn==1.7.2
joblib==1.4.2
scipy>=1.11
```

Molecular docking (GNINA 1.3.2), protein preparation (YASARA 25.1.13), ligand preparation (Open Babel 3.1.1), grid definition (AutoDock Tools 1.5.7) and decoy generation (LUDe) are performed outside this environment and are not installable through pip.

## Citation

If you use this code or data, please cite:

> Tomarchio, E.G., Buccheri, R., Rescifina, A. *Beyond the Score: Fixed-Budget Benchmarking of Virtual Screening Integration Strategies for Decision-Centric Drug Discovery.* Int. J. Mol. Sci., submitted.

*(DOI to be added upon publication.)*

The bioactivity data underlying the deposited datasets are derived from ChEMBL (release 36, https://www.ebi.ac.uk/chembl), which should be cited alongside this work.

## License

The code in this repository is released under the MIT License — see [`LICENSE`](LICENSE).

The datasets under `database/` and the compound-level data in `Beyond_the_Score_VS_results_SI_IJMS.xlsx` are derived from ChEMBL, which is distributed under the Creative Commons Attribution-ShareAlike 3.0 Unported License. They are therefore redistributed here under the same CC BY-SA 3.0 licence — see [`LICENSE-DATA`](LICENSE-DATA).

## Contact
Elisabetta Grazia Tomarchio — Department of Drug and Health Sciences, University of Catania
elisabetta.tomarchio@phd.unict.it 

