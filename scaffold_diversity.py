# -*- coding: utf-8 -*-

"""
SIMILARITY-STRATIFIED VIRTUAL SCREENING ANALYSIS
=================================================

Reviewer-requested robustness analysis.

For each target:
1. Load the benchmark VS library from docking, MCS and QSAR results.
2. Identify the 50 true VS actives.
3. Load the corresponding QSAR training metadata.
4. Calculate Morgan/ECFP4 (radius=2, 2048 bits) nearest-neighbor
   Tanimoto similarity of every VS active against TRAINING ACTIVES.
5. Divide the 50 VS actives into target-specific Low/Mid/High
   similarity terciles (~16-17 compounds per bin).
6. Apply the same fixed-budget selection rule used in the benchmark:
      B = floor(N * cutoff)
   at 1%, 5% and 10%.
7. Evaluate Single Docking, Single MCS and Single ML-QSAR.
8. Pool counts across the five targets.
9. Compare Low vs High similarity bins using Fisher's exact test.
10. Export detailed and manuscript-ready CSV files.

Outputs
-------
SIMILARITY_STRATIFICATION/
    VS_ACTIVE_SIMILARITY_ASSIGNMENTS.csv
    SIMILARITY_STRATIFICATION_PER_TARGET.csv
    SIMILARITY_STRATIFICATION_POOLED.csv
    SIMILARITY_STRATIFICATION_MANUSCRIPT_TABLE.csv
    SIMILARITY_STRATIFICATION_QC.csv
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# USER SETTINGS
# ============================================================

BASE_DIR = "/path/to/project_folder"

EXPECTED_ACTIVES_PER_TARGET = 50

CUTOFFS = {
    "1.0%": 0.01,
    "5.0%": 0.05,
    "10.0%": 0.10
}

# ------------------------------------------------------------
# ACTUAL TRAINING METADATA PATHS PROVIDED BY USER
# ------------------------------------------------------------


TRAIN_CSV_PATHS = {
    "AChE":
        "/path/to/project_folder"
        "training/ACHE/qsar_pipeline_results/train_metadata.csv",

    "EGFR":
        "/path/to/project_folder"
        "training/EGFR/qsar_pipeline_results/train_metadata.csv",

    "hERG":
        "/path/to/project_folder"
        "training/hERG/qsar_pipeline_results/train_metadata.csv",

    "HIV1-P":
        "/path/to/project_folder"
        "training/HIV/qsar_pipeline_results/train_metadata.csv",

    "PPARgamma":
        "/path/to/project_folder"
        "training/PPAR/qsar_pipeline_results/train_metadata.csv",
}

# ------------------------------------------------------------
# ACTUAL TARGET DIRECTORY NAMES
# Used for docking / MCS / QSAR benchmark files
# ------------------------------------------------------------

TARGET_DIRS = {
    "AChE": "ACHE",
    "EGFR": "EGFR",
    "hERG": "hERG",
    "HIV1-P": "HIV",
    "PPARgamma": "PPAR",
}


# ============================================================
# IMPORT RDKit
# ============================================================

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True

except ImportError:
    RDKIT_AVAILABLE = False
    raise ImportError(
        "RDKit is required for ECFP4/Morgan similarity analysis."
    )


# ============================================================
# IMPORT SCIPY
# ============================================================

try:
    from scipy.stats import fisher_exact

    SCIPY_AVAILABLE = True

except ImportError:
    SCIPY_AVAILABLE = False
    raise ImportError(
        "SciPy is required for Fisher's exact test."
    )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def canonicalize_smiles(smiles):
    """
    Canonicalize SMILES while retaining stereochemistry.
    """

    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return smiles

        return Chem.MolToSmiles(
            mol,
            isomericSmiles=True
        )

    except Exception:
        return smiles


def find_column(df, candidates, description):
    """
    Find the first matching column from a list of candidates.
    """

    for col in candidates:
        if col in df.columns:
            return col

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for col in candidates:
        if str(col).lower() in lower_map:
            return lower_map[str(col).lower()]

    raise KeyError(
        f"Could not identify {description}.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def load_csv(path):
    """
    Read CSV using automatic separator detection.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return pd.read_csv(
        path,
        sep=None,
        engine="python"
    )


# ============================================================
# MORGAN / ECFP4
# ============================================================

def get_morgan_fp(smiles):
    """
    ECFP4-equivalent Morgan fingerprint:
        radius = 2
        nBits = 2048
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )


def nearest_training_active_similarity(
    vs_smiles,
    training_active_fps
):
    """
    Maximum ECFP4 Tanimoto similarity between one VS compound
    and all training-set active compounds.
    """

    fp = get_morgan_fp(vs_smiles)

    if fp is None:
        return np.nan

    if not training_active_fps:
        return np.nan

    similarities = DataStructs.BulkTanimotoSimilarity(
        fp,
        training_active_fps
    )

    return float(max(similarities))


# ============================================================
# FIXED-BUDGET SELECTION
# ============================================================

def top_b_by_score(score, B, ascending=False):
    """
    Deterministic top-B selection.

    For all three individual methods in this benchmark,
    higher score = better.
    """

    score = np.asarray(score, dtype=float)

    if ascending:
        order = np.argsort(
            score,
            kind="mergesort"
        )
    else:
        order = np.argsort(
            -score,
            kind="mergesort"
        )

    return order[:B]


# ============================================================
# TARGET-SPECIFIC TERCILES
# ============================================================

def assign_similarity_terciles(df):
    """
    Assign Low / Mid / High similarity bins separately within
    each target.

    Compounds are sorted by nearest-neighbor similarity and split
    into three approximately equal groups.

    For 50 actives:
        Low  = 16 or 17
        Mid  = 16 or 17
        High = 16 or 17
    """

    df = df.sort_values(
        ["Nearest_TrainActive_Tc", "SMILES"],
        ascending=[True, True],
        kind="mergesort"
    ).reset_index(drop=True)

    n = len(df)

    # Deterministic approximately equal thirds.
    bins = np.array_split(
        np.arange(n),
        3
    )

    df["Similarity_Tercile"] = ""

    df.loc[bins[0], "Similarity_Tercile"] = "Low"
    df.loc[bins[1], "Similarity_Tercile"] = "Mid"
    df.loc[bins[2], "Similarity_Tercile"] = "High"

    return df


# ============================================================
# PATHS
# ============================================================

def get_target_paths(target):
    """
    Construct the benchmark file paths using the actual target
    directory names.
    """

    target_dir = TARGET_DIRS[target]

    docking_path = os.path.join(
        BASE_DIR,
        "training",
        "docking",
        f"Enrichment_{target_dir}.csv"
    )

    mcs_path = os.path.join(
        BASE_DIR,
        "training",
        "graph analysis",
        f"{target_dir}_graph.csv"
    )

    qsar_path = os.path.join(
        BASE_DIR,
        "training",
        target_dir,
        "qsar_pipeline_results",
        "VS_QSAR_predictions.csv"
    )

    return docking_path, mcs_path, qsar_path


# ============================================================
# LOAD VS DATA
# ============================================================

def load_vs_library(target):
    """
    Load docking, MCS and QSAR data and merge on canonical SMILES.

    Returns:
        df_merged
    """

    docking_path, mcs_path, qsar_path = get_target_paths(target)

    print("\n" + "=" * 70)
    print(f"TARGET: {target}")
    print("=" * 70)

    print("\nDocking:")
    print(docking_path)

    print("\nMCS:")
    print(mcs_path)

    print("\nQSAR:")
    print(qsar_path)

    # --------------------------------------------------------
    # DOCKING
    # --------------------------------------------------------

    df_dock = load_csv(docking_path)

    smiles_col = find_column(
        df_dock,
        ["smiles", "SMILES"],
        "SMILES column in docking file"
    )

    label_col = find_column(
        df_dock,
        [
            "exp_active",
            "active",
            "label",
            "Activity",
            "activity"
        ],
        "activity label in docking file"
    )

    docking_score_col = find_column(
        df_dock,
        ["CNN_VS"],
        "CNN_VS docking score"
    )

    df_dock = df_dock[
        [smiles_col, label_col, docking_score_col]
    ].copy()

    df_dock.columns = [
        "SMILES",
        "Active",
        "Docking_Score"
    ]

    df_dock["SMILES"] = (
        df_dock["SMILES"]
        .apply(canonicalize_smiles)
    )

    df_dock = (
        df_dock
        .drop_duplicates(
            subset="SMILES",
            keep="first"
        )
    )

    # --------------------------------------------------------
    # MCS
    # --------------------------------------------------------

    df_mcs = load_csv(mcs_path)

    smiles_col = find_column(
        df_mcs,
        ["smiles", "SMILES"],
        "SMILES column in MCS file"
    )

    mcs_score_col = find_column(
        df_mcs,
        [
            "consensus_max_score",
            "Score_MCS"
        ],
        "MCS score"
    )

    df_mcs = df_mcs[
        [smiles_col, mcs_score_col]
    ].copy()

    df_mcs.columns = [
        "SMILES",
        "MCS_Score"
    ]

    df_mcs["SMILES"] = (
        df_mcs["SMILES"]
        .apply(canonicalize_smiles)
    )

    df_mcs = (
        df_mcs
        .drop_duplicates(
            subset="SMILES",
            keep="first"
        )
    )

    # --------------------------------------------------------
    # QSAR
    # --------------------------------------------------------

    df_qsar = load_csv(qsar_path)

    smiles_col = find_column(
        df_qsar,
        ["smiles", "SMILES"],
        "SMILES column in QSAR file"
    )

    qsar_score_col = find_column(
        df_qsar,
        [
            "probability",
            "Score_ML"
        ],
        "QSAR probability"
    )

    df_qsar = df_qsar[
        [smiles_col, qsar_score_col]
    ].copy()

    df_qsar.columns = [
        "SMILES",
        "QSAR_Score"
    ]

    df_qsar["SMILES"] = (
        df_qsar["SMILES"]
        .apply(canonicalize_smiles)
    )

    df_qsar = (
        df_qsar
        .drop_duplicates(
            subset="SMILES",
            keep="first"
        )
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    df = pd.merge(
        df_dock,
        df_mcs,
        on="SMILES",
        how="inner"
    )

    df = pd.merge(
        df,
        df_qsar,
        on="SMILES",
        how="inner"
    )

    df = df.reset_index(drop=True)

    print("\nMerged VS library:")
    print(f"  N = {len(df)}")

    print(
        f"  Active compounds = "
        f"{int((df['Active'] == 1).sum())}"
    )

    if int((df["Active"] == 1).sum()) != EXPECTED_ACTIVES_PER_TARGET:
        raise RuntimeError(
            f"{target}: expected "
            f"{EXPECTED_ACTIVES_PER_TARGET} actives, "
            f"found {int((df['Active'] == 1).sum())}."
        )

    return df


# ============================================================
# LOAD TRAINING ACTIVES
# ============================================================

def load_training_active_fps(target):
    """
    Load training metadata for a target and return Morgan
    fingerprints of training-set ACTIVE compounds.

    The training path is taken directly from TRAIN_CSV_PATHS.
    """

    train_path = TRAIN_CSV_PATHS[target]

    print("\nTraining metadata:")
    print(train_path)

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training metadata not found:\n{train_path}"
        )

    df_train = load_csv(train_path)

    print(
        "Training metadata columns:"
    )
    print(
        list(df_train.columns)
    )

    smiles_col = find_column(
        df_train,
        [
            "smiles",
            "SMILES"
        ],
        "SMILES column in training metadata"
    )

    label_col = find_column(
        df_train,
        [
            "exp_active",
            "active",
            "label",
            "Activity",
            "activity"
        ],
        "activity label in training metadata"
    )

    df_train = df_train[
        [smiles_col, label_col]
    ].copy()

    df_train["SMILES"] = (
        df_train[smiles_col]
        .apply(canonicalize_smiles)
    )

    df_train["Active"] = pd.to_numeric(
        df_train[label_col],
        errors="coerce"
    )

    # --------------------------------------------------------
    # TRAINING ACTIVES ONLY
    # --------------------------------------------------------

    train_actives = df_train[
        df_train["Active"] == 1
    ].copy()

    train_actives = (
        train_actives
        .dropna(subset=["SMILES"])
        .drop_duplicates(
            subset="SMILES"
        )
    )

    print(
        f"Training compounds: {len(df_train)}"
    )

    print(
        f"Training actives: {len(train_actives)}"
    )

    # --------------------------------------------------------
    # FINGERPRINTS
    # --------------------------------------------------------

    fps = []

    invalid = 0

    for smiles in train_actives["SMILES"]:

        fp = get_morgan_fp(smiles)

        if fp is None:
            invalid += 1
            continue

        fps.append(fp)

    print(
        f"Valid training-active fingerprints: {len(fps)}"
    )

    if invalid:
        print(
            f"WARNING: {invalid} training actives "
            f"could not be fingerprinted."
        )

    if len(fps) == 0:
        raise RuntimeError(
            f"No valid training active fingerprints for {target}."
        )

    return fps


# ============================================================
# SIMILARITY ANALYSIS FOR ONE TARGET
# ============================================================

def calculate_target_similarity(
    target,
    df_vs,
    training_active_fps
):
    """
    Calculate nearest-neighbor similarity for all 50 VS actives.
    """

    vs_actives = df_vs[
        df_vs["Active"] == 1
    ].copy()

    if len(vs_actives) != EXPECTED_ACTIVES_PER_TARGET:
        raise RuntimeError(
            f"{target}: expected "
            f"{EXPECTED_ACTIVES_PER_TARGET} VS actives, "
            f"found {len(vs_actives)}."
        )

    similarities = []

    print(
        f"\nCalculating nearest-neighbor "
        f"ECFP4 similarity for {len(vs_actives)} actives..."
    )

    for _, row in vs_actives.iterrows():

        tc = nearest_training_active_similarity(
            row["SMILES"],
            training_active_fps
        )

        similarities.append(tc)

    vs_actives["Nearest_TrainActive_Tc"] = similarities

    if vs_actives[
        "Nearest_TrainActive_Tc"
    ].isna().any():

        raise RuntimeError(
            f"{target}: missing similarity values."
        )

    # --------------------------------------------------------
    # TERCILES
    # --------------------------------------------------------

    vs_actives = assign_similarity_terciles(
        vs_actives
    )

    print("\nSimilarity summary:")

    print(
        vs_actives[
            "Nearest_TrainActive_Tc"
        ].describe()
    )

    print("\nTercile counts:")

    print(
        vs_actives[
            "Similarity_Tercile"
        ].value_counts()
        .reindex(["Low", "Mid", "High"])
    )

    print("\nTercile boundaries:")

    for tercile in ["Low", "Mid", "High"]:

        values = vs_actives.loc[
            vs_actives["Similarity_Tercile"] == tercile,
            "Nearest_TrainActive_Tc"
        ]

        print(
            f"  {tercile}: "
            f"n={len(values)}, "
            f"min={values.min():.3f}, "
            f"median={values.median():.3f}, "
            f"max={values.max():.3f}"
        )

    return vs_actives


# ============================================================
# PER-TARGET SCREENING
# ============================================================

def evaluate_target(
    target,
    df_vs,
    active_similarity_df
):
    """
    Apply fixed-budget top-B selection to the three individual
    screening methods.
    """

    N = len(df_vs)

    # Map SMILES -> similarity tercile
    similarity_map = (
        active_similarity_df
        .set_index("SMILES")[
            "Similarity_Tercile"
        ]
        .to_dict()
    )

    similarity_tc_map = (
        active_similarity_df
        .set_index("SMILES")[
            "Nearest_TrainActive_Tc"
        ]
        .to_dict()
    )

    # Add similarity information to VS library
    df_vs = df_vs.copy()

    df_vs["Similarity_Tercile"] = (
        df_vs["SMILES"]
        .map(similarity_map)
    )

    df_vs["Nearest_TrainActive_Tc"] = (
        df_vs["SMILES"]
        .map(similarity_tc_map)
    )

    # --------------------------------------------------------
    # SCORE DEFINITIONS
    # --------------------------------------------------------

    methods = {
        "Docking": "Docking_Score",
        "MCS": "MCS_Score",
        "ML-QSAR": "QSAR_Score"
    }

    rows = []

    for cutoff_label, cutoff in CUTOFFS.items():

        B = int(
            np.floor(
                N * cutoff
            )
        )

        B = max(
            1,
            B
        )

        print(
            f"\n{target} | {cutoff_label}: "
            f"N={N}, B={B}"
        )

        for method, score_col in methods.items():

            scores = (
                df_vs[score_col]
                .astype(float)
                .values
            )

            selected_idx = top_b_by_score(
                scores,
                B,
                ascending=False
            )

            selected = df_vs.iloc[
                selected_idx
            ].copy()

            # ------------------------------------------------
            # Only true actives can contribute to
            # similarity-stratified recall.
            # ------------------------------------------------

            selected_actives = selected[
                selected["Active"] == 1
            ].copy()

            for tercile in [
                "Low",
                "Mid",
                "High"
            ]:

                total_bin = int(
                    (
                        active_similarity_df[
                            "Similarity_Tercile"
                        ] == tercile
                    ).sum()
                )

                recovered_bin = int(
                    (
                        selected_actives[
                            "Similarity_Tercile"
                        ] == tercile
                    ).sum()
                )

                recall = (
                    100.0 * recovered_bin / total_bin
                    if total_bin > 0
                    else np.nan
                )

                rows.append({
                    "Target": target,
                    "Cutoff": cutoff_label,
                    "Method": method,
                    "Budget_B": B,
                    "Similarity_Tercile": tercile,
                    "N_Actives_in_Tercile": total_bin,
                    "Recovered_Actives": recovered_bin,
                    "Recall_percent": recall
                })

            # ------------------------------------------------
            # Save selected actives for detailed QC
            # ------------------------------------------------

    return pd.DataFrame(rows)


# ============================================================
# FISHER EXACT TEST
# ============================================================

def fisher_low_high(
    pooled_df,
    cutoff,
    method
):
    """
    Compare recovered vs non-recovered actives between
    Low- and High-similarity bins.

    2 x 2 table:

                  recovered    not recovered
    Low              a             b
    High             c             d
    """

    subset = pooled_df[
        (pooled_df["Cutoff"] == cutoff)
        &
        (pooled_df["Method"] == method)
    ].copy()

    low = subset[
        subset["Similarity_Tercile"] == "Low"
    ].iloc[0]

    high = subset[
        subset["Similarity_Tercile"] == "High"
    ].iloc[0]

    a = int(low["Recovered_Actives"])
    b = int(
        low["N_Actives_in_Tercile"] - a
    )

    c = int(high["Recovered_Actives"])
    d = int(
        high["N_Actives_in_Tercile"] - c
    )

    table = np.array([
        [a, b],
        [c, d]
    ])

    odds_ratio, p_value = fisher_exact(
        table,
        alternative="two-sided"
    )

    return {
        "Low_Recovered": a,
        "Low_Total": a + b,
        "High_Recovered": c,
        "High_Total": c + d,
        "Low_Recall_percent": (
            100 * a / (a + b)
        ),
        "High_Recall_percent": (
            100 * c / (c + d)
        ),
        "Odds_Ratio": odds_ratio,
        "Fisher_p": p_value
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_similarity_stratification():

    output_dir = os.path.join(
        BASE_DIR,
        "SIMILARITY_STRATIFICATION"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    all_similarity = []
    all_results = []
    qc_rows = []

    targets = [
        "AChE",
        "EGFR",
        "hERG",
        "HIV1-P",
        "PPARgamma"
    ]

    # ========================================================
    # TARGET LOOP
    # ========================================================

    for target in targets:

        try:

            # ------------------------------------------------
            # Load VS benchmark library
            # ------------------------------------------------

            df_vs = load_vs_library(
                target
            )

            # ------------------------------------------------
            # Load training active fingerprints
            # ------------------------------------------------

            train_active_fps = (
                load_training_active_fps(
                    target
                )
            )

            # ------------------------------------------------
            # Calculate similarity and terciles
            # ------------------------------------------------

            active_similarity = (
                calculate_target_similarity(
                    target,
                    df_vs,
                    train_active_fps
                )
            )

            active_similarity[
                "Target"
            ] = target

            # ------------------------------------------------
            # Reorder columns
            # ------------------------------------------------

            active_similarity = active_similarity[
                [
                    "Target",
                    "SMILES",
                    "Nearest_TrainActive_Tc",
                    "Similarity_Tercile"
                ]
            ]

            all_similarity.append(
                active_similarity
            )

            # ------------------------------------------------
            # Evaluate fixed-budget retrieval
            # ------------------------------------------------

            target_results = evaluate_target(
                target,
                df_vs,
                active_similarity
            )

            all_results.append(
                target_results
            )

            # ------------------------------------------------
            # QC
            # ------------------------------------------------

            for tercile in [
                "Low",
                "Mid",
                "High"
            ]:

                subset = active_similarity[
                    active_similarity[
                        "Similarity_Tercile"
                    ] == tercile
                ]

                qc_rows.append({
                    "Target": target,
                    "Similarity_Tercile": tercile,
                    "N": len(subset),
                    "Min_Tc": subset[
                        "Nearest_TrainActive_Tc"
                    ].min(),
                    "Median_Tc": subset[
                        "Nearest_TrainActive_Tc"
                    ].median(),
                    "Max_Tc": subset[
                        "Nearest_TrainActive_Tc"
                    ].max()
                })

        except Exception as e:

            print(
                "\nERROR processing "
                f"{target}: {e}"
            )

            qc_rows.append({
                "Target": target,
                "Similarity_Tercile": "ERROR",
                "N": np.nan,
                "Min_Tc": np.nan,
                "Median_Tc": np.nan,
                "Max_Tc": np.nan,
                "Error": str(e)
            })

            raise

    # ========================================================
    # COMBINE DATA
    # ========================================================

    df_similarity = pd.concat(
        all_similarity,
        ignore_index=True
    )

    df_results = pd.concat(
        all_results,
        ignore_index=True
    )

    df_qc = pd.DataFrame(
        qc_rows
    )

    # ========================================================
    # SAVE PER-ACTIVE SIMILARITY
    # ========================================================

    similarity_path = os.path.join(
        output_dir,
        "VS_ACTIVE_SIMILARITY_ASSIGNMENTS.csv"
    )

    df_similarity.to_csv(
        similarity_path,
        index=False
    )

    # ========================================================
    # SAVE PER-TARGET RESULTS
    # ========================================================

    per_target_path = os.path.join(
        output_dir,
        "SIMILARITY_STRATIFICATION_PER_TARGET.csv"
    )

    df_results.to_csv(
        per_target_path,
        index=False
    )

    # ========================================================
    # POOLED COUNTS
    # ========================================================

    pooled = (
        df_results
        .groupby(
            [
                "Cutoff",
                "Method",
                "Similarity_Tercile"
            ],
            as_index=False
        )
        .agg(
            N_Actives_in_Tercile=(
                "N_Actives_in_Tercile",
                "sum"
            ),
            Recovered_Actives=(
                "Recovered_Actives",
                "sum"
            )
        )
    )

    pooled["Recall_percent"] = (
        100.0
        * pooled["Recovered_Actives"]
        / pooled["N_Actives_in_Tercile"]
    )

    pooled_path = os.path.join(
        output_dir,
        "SIMILARITY_STRATIFICATION_POOLED.csv"
    )

    pooled.to_csv(
        pooled_path,
        index=False
    )

    # ========================================================
    # FISHER TESTS
    # ========================================================

    fisher_rows = []

    for cutoff in CUTOFFS.keys():

        for method in [
            "Docking",
            "MCS",
            "ML-QSAR"
        ]:

            stats = fisher_low_high(
                pooled,
                cutoff,
                method
            )

            fisher_rows.append({
                "Cutoff": cutoff,
                "Method": method,
                **stats
            })

    fisher_df = pd.DataFrame(
        fisher_rows
    )

    # ========================================================
    # MANUSCRIPT-READY TABLE
    # ========================================================

    manuscript_rows = []

    for _, row in fisher_df.iterrows():

        manuscript_rows.append({
            "Cutoff": row["Cutoff"],
            "Method": row["Method"],
            "Low-similarity recall":
                f"{row['Low_Recovered']}/"
                f"{row['Low_Total']} "
                f"({row['Low_Recall_percent']:.1f}%)",
            "High-similarity recall":
                f"{row['High_Recovered']}/"
                f"{row['High_Total']} "
                f"({row['High_Recall_percent']:.1f}%)",
            "Fisher_p":
                row["Fisher_p"],
            "Odds_Ratio":
                row["Odds_Ratio"]
        })

    manuscript_df = pd.DataFrame(
        manuscript_rows
    )

    manuscript_path = os.path.join(
        output_dir,
        "SIMILARITY_STRATIFICATION_MANUSCRIPT_TABLE.csv"
    )

    manuscript_df.to_csv(
        manuscript_path,
        index=False
    )

    # ========================================================
    # SAVE FISHER DETAILS
    # ========================================================

    fisher_path = os.path.join(
        output_dir,
        "SIMILARITY_STRATIFICATION_FISHER_TESTS.csv"
    )

    fisher_df.to_csv(
        fisher_path,
        index=False
    )

    # ========================================================
    # SAVE QC
    # ========================================================

    qc_path = os.path.join(
        output_dir,
        "SIMILARITY_STRATIFICATION_QC.csv"
    )

    df_qc.to_csv(
        qc_path,
        index=False
    )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 90)
    print("SIMILARITY-STRATIFICATION RESULTS")
    print("=" * 90)

    print(
        "\nPooled Low-vs-High similarity comparison:"
    )

    print()

    for _, row in manuscript_df.iterrows():

        print(
            f"{row['Cutoff']:>5} | "
            f"{row['Method']:<10} | "
            f"Low = "
            f"{row['Low-similarity recall']:<16} | "
            f"High = "
            f"{row['High-similarity recall']:<16} | "
            f"p = "
            f"{row['Fisher_p']:.3e}"
        )

    print("\n")
    print("=" * 90)
    print("OUTPUT FILES")
    print("=" * 90)

    print(
        "\nActive-level similarity assignments:"
    )
    print(similarity_path)

    print(
        "\nPer-target stratification:"
    )
    print(per_target_path)

    print(
        "\nPooled results:"
    )
    print(pooled_path)

    print(
        "\nManuscript-ready table:"
    )
    print(manuscript_path)

    print(
        "\nFisher exact tests:"
    )
    print(fisher_path)

    print(
        "\nQC:"
    )
    print(qc_path)

    print("\nAnalysis completed successfully.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_similarity_stratification()
