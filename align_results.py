# -*- coding: utf-8 -*-
"""
Benchmark aggregation and statistical/scaffold-diversity analysis for
the fixed-budget virtual screening study.

The script:
1. merges docking, MCS, and ML-QSAR predictions;
2. evaluates the 20 predefined screening strategies;
3. calculates fixed-budget recall, precision, and TP counts;
4. performs a Monte Carlo permutation Friedman omnibus test;
5. calculates Bemis-Murcko scaffold diversity metrics;
6. calculates pairwise scaffold Jaccard similarity between strategies;
7. performs QC checks on compound/active retention.

The analysis is performed independently for the five benchmark targets
and for 1%, 5%, and 10% screening budgets.
"""
# Outputs generated:
#  - SINGLE_TARGET_PERFORMANCE_3M.csv      (Performance data per target/cutoff/strategy)
#  - FINAL_STATISTICAL_VALIDATION_3M.csv    (Mean ± SD across targets)
#  - FRIEDMAN_OMNIBUS_COMPARISON.csv       (Omnibus test across strategies, per cutoff)
#  - QC_REPORT.csv                         (Summary QC diagnostics)
#  - SCAFFOLD_METRICS_PER_STRATEGY.csv     (Scaffold diversity and SCR per target/cutoff/strategy)
#  - TOP_SCAFFOLDS_BY_STRATEGY.csv         (Detailed breakdown of top scaffolds per selection)
#  - SCAFFOLD_OVERLAP_STRATEGIES.csv       (Pairwise scaffold overlap/Jaccard similarity between strategies)

import os
import glob
import pandas as pd
import numpy as np
from itertools import combinations

EXPECTED_ACTIVES_PER_TARGET = 50      
KEEP_STEREOCHEMISTRY = True           
PREFILTER_MULTIPLIER_2STAGE = 2       
PREFILTER_MULTIPLIER_3STAGE_1 = 3     
PREFILTER_MULTIPLIER_3STAGE_2 = 2     

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
    print("RDKit detected: SMILES canonicalization and Bemis-Murcko scaffold generation enabled.")
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not found. Scaffold analysis cannot be executed without RDKit.")

try:
    from scipy.stats import friedmanchisquare
    from numpy.random import default_rng
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not found. The Friedman omnibus test will be skipped.")


def canonicalize_smiles(smiles_str):
    """Converts a SMILES string to its canonical form."""
    if not RDKIT_AVAILABLE or pd.isna(smiles_str):
        return str(smiles_str).strip()
    try:
        mol = Chem.MolFromSmiles(str(smiles_str).strip())
        if mol:
            if not KEEP_STEREOCHEMISTRY:
                Chem.RemoveStereochemistry(mol)
            return Chem.MolToSmiles(mol, isomericSmiles=KEEP_STEREOCHEMISTRY)
    except Exception:
        pass
    return str(smiles_str).strip()


def get_bemis_murcko_scaffold(smiles_str):
    """Generates the canonical Bemis-Murcko scaffold for a given SMILES string."""
    if not RDKIT_AVAILABLE or pd.isna(smiles_str):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles_str).strip())
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except Exception:
        return None


def dedupe_with_label_check(df, label_col, source_name, qc_log):
    """Removes duplicate SMILES, dropping inconsistent active/decoy labels entirely."""
    if label_col not in df.columns:
        dup_mask = df.duplicated(subset=['smiles'], keep=False)
        if dup_mask.any():
            qc_log.append({
                'source': source_name, 'issue': 'duplicate_smiles_no_label',
                'n_duplicate_rows': int(dup_mask.sum())
            })
        return df.drop_duplicates(subset=['smiles'], keep='first').copy()

    grouped = df.groupby('smiles')[label_col].nunique()
    inconsistent = grouped[grouped > 1].index.tolist()
    if inconsistent:
        qc_log.append({
            'source': source_name, 'issue': 'inconsistent_label_for_duplicate_smiles',
            'n_conflicting_smiles': len(inconsistent),
            'example_smiles': inconsistent[:5]
        })
        df = df[~df['smiles'].isin(inconsistent)].copy()
    return df.drop_duplicates(subset=['smiles'], keep='first').copy()


def calculate_metrics(selected_mask, true_labels, actives_denominator):
    """Calculates recall and precision as percentages."""
    true_positives = int(np.sum(selected_mask & (true_labels == 1)))
    total_selected = int(np.sum(selected_mask))
    recall = (true_positives / actives_denominator) * 100 if actives_denominator > 0 else 0.0
    precision = (true_positives / total_selected) * 100 if total_selected > 0 else 0.0
    return recall, precision, true_positives, total_selected


def top_b_by_score(score, B, ascending=False):
    """Returns indices of the top B elements."""
    order = np.argsort(score if ascending else -score, kind='mergesort')
    return order[:B]


def rank_min(score):
    """Rank 1 = best, ties handled using 'min' method."""
    order = np.argsort(-score, kind='mergesort')
    ranks = np.empty(len(score), dtype=float)
    sorted_scores = score[order]
    r = 1
    i = 0
    n = len(score)
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = r
        r += (j - i)
        i = j
    return ranks


def sequential_funnel(scores_stage_order, B, prefilter_sizes, N):
    """Sequential funnel pre-filtering and re-ranking."""
    survivors = np.arange(N)
    for stage_idx, score in enumerate(scores_stage_order[:-1]):
        P = min(len(survivors), max(B, prefilter_sizes[stage_idx]))
        stage_scores = score[survivors]
        top_local = top_b_by_score(stage_scores, P, ascending=False)
        survivors = survivors[top_local]
    last_score = scores_stage_order[-1][survivors]
    final_local = top_b_by_score(last_score, min(B, len(survivors)), ascending=False)
    return survivors[final_local]


def friedman_permutation_test(pivot, n_perm=100000, seed=42):
    """Monte Carlo permutation Friedman test."""
    rng = default_rng(seed)
    data = pivot.values
    n_blocks, n_treatments = data.shape
    chi2, _ = friedmanchisquare(*[data[:, j] for j in range(n_treatments)])
    exceed = 0

    for _ in range(n_perm):
        perm = data.copy()
        for i in range(n_blocks):
            perm[i] = rng.permutation(perm[i])
        stat, _ = friedmanchisquare(*[perm[:, j] for j in range(n_treatments)])
        if stat >= chi2:
            exceed += 1

    p_mc = (exceed + 1) / (n_perm + 1)
    return chi2, p_mc


def run_friedman_omnibus(df_all, output_dir, cutoff_labels):
    """Omnibus Friedman rank-sum test across screening strategies."""
    if not SCIPY_AVAILABLE:
        print("Friedman omnibus test skipped: SciPy is not installed.\n")
        return None

    rows = []
    print("Friedman omnibus test (blocks = targets, response = recovered actives)")

    for c_lbl in cutoff_labels:
        pivot = df_all[df_all['Cutoff'] == c_lbl].pivot_table(
            index='Target', columns='Strategy', values='TP')
        n_incomplete = int(pivot.isna().any(axis=1).sum())
        pivot = pivot.dropna(axis=0, how='any')
        n_blocks, n_strategies = pivot.shape

        if n_blocks < 3 or n_strategies < 3:
            continue

        chi2, p_value = friedman_permutation_test(pivot, n_perm=100000, seed=42)
        mean_rank = pivot.rank(axis=1, ascending=False, method='average').mean(axis=0)
        best_strategy = mean_rank.idxmin()

        note = ""
        if n_incomplete:
            note = f"{n_incomplete} target(s) dropped for incomplete coverage; "
        if n_blocks < 10:
            note += "Monte Carlo permutation p-value (100000 permutations, seed=42)."

        rows.append({
            'Cutoff': c_lbl,
            'N_targets': n_blocks,
            'N_strategies': n_strategies,
            'df': n_strategies - 1,
            'Chi2': round(float(chi2), 3),
            'p_value': float(p_value),
            'Statistic': 'Friedman chi-square',
            'Permutation_test': True,
            'Permutations': 100000,
            'Random_seed': 42,
            'Best_strategy_by_mean_rank': best_strategy,
            'Best_mean_rank': round(float(mean_rank.min()), 3),
            'Note': note.strip()
        })

    if not rows:
        return None

    friedman_path = os.path.join(output_dir, "FRIEDMAN_OMNIBUS_STRATEGY_COMPARISON.csv")
    pd.DataFrame(rows).to_csv(friedman_path, index=False)
    return friedman_path


def analyze_scaffolds_for_selection(selected_df, target_label_col):
    """Computes scaffold summary metrics and top scaffolds for a selected subset."""
    scaffolds = selected_df['scaffold'].dropna()
    n_selected = len(selected_df)
    n_valid = len(scaffolds)

    if n_valid == 0:
        summary = {
            'N_selected': n_selected,
            'N_valid_scaffolds': 0,
            'N_unique_scaffolds': 0,
            'SCR': np.nan,
            'Singleton_fraction': np.nan,
            'Max_compounds_per_scaffold': np.nan
        }
        return summary, pd.DataFrame()

    counts = scaffolds.value_counts()
    n_unique = len(counts)
    n_singletons = int((counts == 1).sum())

    summary = {
        'N_selected': n_selected,
        'N_valid_scaffolds': n_valid,
        'N_unique_scaffolds': n_unique,
        'SCR': n_unique / n_selected if n_selected > 0 else np.nan,
        'Singleton_fraction': n_singletons / n_unique if n_unique > 0 else np.nan,
        'Max_compounds_per_scaffold': int(counts.max())
    }

    # Detailed top scaffolds aggregation
    top_scaffolds = (
        selected_df.groupby('scaffold')
        .agg(
            Total_Compounds=('smiles', 'count'),
            Active_Compounds=(target_label_col, lambda x: int((x == 1).sum()))
        )
        .reset_index()
        .sort_values(by=['Total_Compounds', 'Active_Compounds'], ascending=False)
    )
    top_scaffolds['Pct_of_Selected'] = (top_scaffolds['Total_Compounds'] / n_selected) * 100

    return summary, top_scaffolds


def build_ultimate_statistical_pipeline(base_dir):
    docking_dir = os.path.join(base_dir, "training/docking")
    docking_files = glob.glob(os.path.join(docking_dir, "Enrichment_*.csv"))

    if not docking_files:
        raise FileNotFoundError(
            f"No files found at path: {docking_dir}\n"
            "Check whether the files are actually located there."
        )

    cutoffs = [0.01, 0.05, 0.10]
    cutoff_labels = ['1.0%', '5.0%', '10.0%']
    all_target_results = []
    qc_log = []

    # Scaffold output collections
    scaffold_metrics_list = []
    top_scaffolds_list = []
    scaffold_overlap_list = []

    print("Starting data merge, performance calculation, and scaffold analysis...\n")

    for dock_path in docking_files:
        target_name = os.path.basename(dock_path).replace("Enrichment_", "").replace(".csv", "")
        print(f"--- Processing Target: {target_name} ---")

        # 1. Load docking file
        df_dock = pd.read_csv(dock_path, sep=None, engine='python')
        df_dock.columns = [c if c.lower() != 'smiles' else 'smiles' for c in df_dock.columns]
        df_dock['smiles'] = df_dock['smiles'].apply(canonicalize_smiles)

        target_label_col = 'exp_active'
        if target_label_col not in df_dock.columns:
            possible_cols = [c for c in df_dock.columns if 'active' in c.lower() or 'label' in c.lower()]
            if len(possible_cols) == 1:
                target_label_col = possible_cols[0]
            else:
                raise KeyError(f"Missing or ambiguous active column in {os.path.basename(dock_path)}")

        n_dock_raw = len(df_dock)
        n_actives_dock_raw = int((df_dock[target_label_col] == 1).sum())

        df_dock_clean = df_dock[['smiles', 'CNN_VS', target_label_col]].copy()
        df_dock_clean = dedupe_with_label_check(df_dock_clean, target_label_col,
                                                 f"{target_name}/docking", qc_log)

        # 2. Dynamic paths for MCS and QSAR
        mcs_path = os.path.join(base_dir, f"training/graph analysis/{target_name}_graph.csv")
        qsar_path = os.path.join(base_dir, f"training/{target_name}/qsar_pipeline_results/VS_QSAR_predictions.csv")

        if not os.path.exists(mcs_path):
            print(f"Skipping target {target_name}: MCS file not found at {mcs_path}")
            qc_log.append({'source': target_name, 'issue': 'missing_mcs_file', 'path': mcs_path})
            continue
        df_mcs = pd.read_csv(mcs_path, sep=None, engine='python')
        df_mcs.columns = [c if c.lower() != 'smiles' else 'smiles' for c in df_mcs.columns]
        df_mcs.rename(columns={'consensus_max_score': 'Score_MCS'}, inplace=True)
        df_mcs['smiles'] = df_mcs['smiles'].apply(canonicalize_smiles)
        df_mcs_clean = df_mcs[['smiles', 'Score_MCS']].copy()
        df_mcs_clean = dedupe_with_label_check(df_mcs_clean, None, f"{target_name}/mcs", qc_log)

        if not os.path.exists(qsar_path):
            print(f"Skipping target {target_name}: QSAR file not found at {qsar_path}")
            qc_log.append({'source': target_name, 'issue': 'missing_qsar_file', 'path': qsar_path})
            continue
        df_qsar = pd.read_csv(qsar_path, sep=None, engine='python')
        df_qsar.columns = [c if c.lower() != 'smiles' else 'smiles' for c in df_qsar.columns]
        df_qsar.rename(columns={'probability': 'Score_ML'}, inplace=True)
        df_qsar['smiles'] = df_qsar['smiles'].apply(canonicalize_smiles)
        df_qsar_clean = df_qsar[['smiles', 'Score_ML']].copy()
        df_qsar_clean = dedupe_with_label_check(df_qsar_clean, None, f"{target_name}/qsar", qc_log)

        # 3. MERGE
        df_merged = pd.merge(df_dock_clean, df_mcs_clean, on='smiles', how='inner')
        df_merged = pd.merge(df_merged, df_qsar_clean, on='smiles', how='inner')

        N_total = len(df_merged)
        if N_total == 0:
            print(f"WARNING: 0 molecules in common after the merge for {target_name}!")
            qc_log.append({'source': target_name, 'issue': 'empty_merge_result'})
            continue

        n_actives_merged = int((df_merged[target_label_col] == 1).sum())
        molecules_lost = n_dock_raw - N_total
        actives_lost = n_actives_dock_raw - n_actives_merged

        qc_log.append({
            'source': target_name, 'issue': 'merge_summary',
            'n_dock_raw': n_dock_raw, 'n_actives_dock_raw': n_actives_dock_raw,
            'N_total_merged': N_total, 'n_actives_merged': n_actives_merged,
            'molecules_lost': molecules_lost, 'actives_lost': actives_lost,
            'qc_pass_50_actives': n_actives_merged == EXPECTED_ACTIVES_PER_TARGET
        })

        df_merged = df_merged.reset_index(drop=True)

        # Pre-calculate Bemis-Murcko scaffolds for all merged compounds
        if RDKIT_AVAILABLE:
            df_merged['scaffold'] = df_merged['smiles'].apply(get_bemis_murcko_scaffold)

        labels = df_merged[target_label_col].values.astype(int)
        actives_denominator = n_actives_merged

        score_qsar = df_merged['Score_ML'].values.astype(float)
        score_mcs = df_merged['Score_MCS'].values.astype(float)
        score_dock = df_merged['CNN_VS'].values.astype(float)

        rank_qsar = rank_min(score_qsar)
        rank_mcs = rank_min(score_mcs)
        rank_dock = rank_min(score_dock)

        # 4. Metrics for 3 cutoffs x 20 strategies
        for c_val, c_lbl in zip(cutoffs, cutoff_labels):
            B = max(1, int(N_total * c_val))

            selections = {}

            # --- Single methods ---
            selections['Single QSAR'] = top_b_by_score(score_qsar, B)
            selections['Single MCS'] = top_b_by_score(score_mcs, B)
            selections['Single Docking'] = top_b_by_score(score_dock, B)

            # --- Rank-fusion strategies ---
            selections['Best-Rank Fusion (QSAR-MCS)'] = top_b_by_score(
                np.minimum(rank_qsar, rank_mcs), B, ascending=True)
            selections['Best-Rank Fusion (3-Method)'] = top_b_by_score(
                np.minimum(np.minimum(rank_qsar, rank_mcs), rank_dock), B, ascending=True)

            selections['Worst-Rank Fusion (QSAR-MCS)'] = top_b_by_score(
                np.maximum(rank_qsar, rank_mcs), B, ascending=True)
            selections['Worst-Rank Fusion (3-Method)'] = top_b_by_score(
                np.maximum(np.maximum(rank_qsar, rank_mcs), rank_dock), B, ascending=True)

            # --- Consensus ---
            mean_rank = (rank_qsar + rank_mcs + rank_dock) / 3.0
            selections['Consensus (Mean Rank 3M)'] = top_b_by_score(mean_rank, B, ascending=True)

            # --- Two-stage funnels ---
            P2 = PREFILTER_MULTIPLIER_2STAGE * B
            two_stage_pairs = [
                ('Sequential (QSAR -> MCS)', score_qsar, score_mcs),
                ('Sequential (MCS -> QSAR)', score_mcs, score_qsar),
                ('Sequential (QSAR -> Docking)', score_qsar, score_dock),
                ('Sequential (Docking -> QSAR)', score_dock, score_qsar),
                ('Sequential (MCS -> Docking)', score_mcs, score_dock),
                ('Sequential (Docking -> MCS)', score_dock, score_mcs),
            ]
            for name, score_a, score_b in two_stage_pairs:
                selections[name] = sequential_funnel(
                    [score_a, score_b], B, prefilter_sizes=[P2], N=N_total)

            # --- Three-stage funnels ---
            P3_1 = PREFILTER_MULTIPLIER_3STAGE_1 * B
            P3_2 = PREFILTER_MULTIPLIER_3STAGE_2 * B
            three_stage_triples = [
                ('Sequential (3-Stage: QSAR -> MCS -> Docking)', score_qsar, score_mcs, score_dock),
                ('Sequential (3-Stage: QSAR -> Docking -> MCS)', score_qsar, score_dock, score_mcs),
                ('Sequential (3-Stage: MCS -> QSAR -> Docking)', score_mcs, score_qsar, score_dock),
                ('Sequential (3-Stage: MCS -> Docking -> QSAR)', score_mcs, score_dock, score_qsar),
                ('Sequential (3-Stage: Docking -> QSAR -> MCS)', score_dock, score_qsar, score_mcs),
                ('Sequential (3-Stage: Docking -> MCS -> QSAR)', score_dock, score_mcs, score_qsar),
            ]
            for name, score_a, score_b, score_c in three_stage_triples:
                selections[name] = sequential_funnel(
                    [score_a, score_b, score_c], B, prefilter_sizes=[P3_1, P3_2], N=N_total)

            # Strategy scaffold tracking dictionary for cross-comparison
            strategy_scaffold_sets = {}

            # 5. Performance and Scaffold Metric Calculation
            for strat_name, idx in selections.items():
                mask = np.zeros(N_total, dtype=bool)
                mask[idx] = True
                rec, prec, tp, n_sel_actual = calculate_metrics(mask, labels, actives_denominator)

                all_target_results.append({
                    'Target': target_name,
                    'Cutoff': c_lbl,
                    'Strategy': strat_name,
                    'Budget_B': B,
                    'Selected_N': n_sel_actual,
                    'TP': tp,
                    'Recall': rec,
                    'Precision': prec
                })

                # Scaffold Analysis for current strategy selection
                if RDKIT_AVAILABLE:
                    selected_df = df_merged.iloc[idx].copy()
                    scaf_summary, top_scafs = analyze_scaffolds_for_selection(selected_df, target_label_col)

                    scaf_summary.update({
                        'Target': target_name,
                        'Cutoff': c_lbl,
                        'Budget_B': B,
                        'Strategy': strat_name
                    })
                    scaffold_metrics_list.append(scaf_summary)

                    if not top_scafs.empty:
                        top_scafs['Target'] = target_name
                        top_scafs['Cutoff'] = c_lbl
                        top_scafs['Budget_B'] = B
                        top_scafs['Strategy'] = strat_name
                        top_scaffolds_list.append(top_scafs)

                        strategy_scaffold_sets[strat_name] = set(top_scafs['scaffold'].dropna().tolist())
                    else:
                        strategy_scaffold_sets[strat_name] = set()

            # Pairwise Scaffold Overlap Analysis across Strategies for this Target and Cutoff
            if RDKIT_AVAILABLE and len(strategy_scaffold_sets) > 1:
                for strat_a, strat_b in combinations(strategy_scaffold_sets.keys(), 2):
                    set_a = strategy_scaffold_sets[strat_a]
                    set_b = strategy_scaffold_sets[strat_b]

                    intersection = len(set_a.intersection(set_b))
                    union = len(set_a.union(set_b))
                    jaccard = (intersection / union) if union > 0 else 0.0

                    scaffold_overlap_list.append({
                        'Target': target_name,
                        'Cutoff': c_lbl,
                        'Budget_B': B,
                        'Strategy_A': strat_a,
                        'Strategy_B': strat_b,
                        'Shared_Scaffolds': intersection,
                        'Union_Scaffolds': union,
                        'Jaccard_Scaffold_Similarity': round(jaccard, 4)
                    })

    if not all_target_results:
        print("Error: no data collected. Check input paths.")
        return None

    # Export Directories and Performance CSVs
    df_all = pd.DataFrame(all_target_results)
    output_dir = os.path.join(base_dir, "GLOBAL_STUDY_OUTPUT")
    os.makedirs(output_dir, exist_ok=True)

    single_target_path = os.path.join(output_dir, "SINGLE_TARGET_PERFORMANCE_3M.csv")
    df_all.to_csv(single_target_path, index=False)

    df_stats = df_all.groupby(['Cutoff', 'Strategy']).agg(
        Recall_mean=('Recall', 'mean'),
        Recall_std=('Recall', 'std'),
        Precision_mean=('Precision', 'mean'),
        Precision_std=('Precision', 'std'),
        Mean_Selected_N=('Selected_N', 'mean')
    ).reset_index().fillna(0.0)

    final_output_path = os.path.join(output_dir, "FINAL_STATISTICAL_VALIDATION_3M.csv")
    df_stats.to_csv(final_output_path, index=False)

    qc_path = os.path.join(output_dir, "QC_REPORT.csv")
    pd.DataFrame(qc_log).to_csv(qc_path, index=False)

    friedman_path = run_friedman_omnibus(df_all, output_dir, cutoff_labels)

    # Export Scaffold Analysis CSVs
    scaf_metrics_path, top_scaf_path, scaf_overlap_path = None, None, None
    if RDKIT_AVAILABLE:
        if scaffold_metrics_list:
            df_scaf_metrics = pd.DataFrame(scaffold_metrics_list)
            scaf_metrics_path = os.path.join(output_dir, "SCAFFOLD_METRICS_PER_STRATEGY.csv")
            df_scaf_metrics.to_csv(scaf_metrics_path, index=False)

        if top_scaffolds_list:
            df_top_scaffolds = pd.concat(top_scaffolds_list, ignore_index=True)
            top_scaf_path = os.path.join(output_dir, "TOP_SCAFFOLDS_BY_STRATEGY.csv")
            df_top_scaffolds.to_csv(top_scaf_path, index=False)

        if scaffold_overlap_list:
            df_scaf_overlap = pd.DataFrame(scaffold_overlap_list)
            scaf_overlap_path = os.path.join(output_dir, "SCAFFOLD_OVERLAP_STRATEGIES.csv")
            df_scaf_overlap.to_csv(scaf_overlap_path, index=False)

    print("=" * 60)
    print("PIPELINE & SCAFFOLD ANALYSIS COMPLETE")
    print(f"Performance raw data:      {single_target_path}")
    print(f"Summary statistics:       {final_output_path}")
    if friedman_path:
        print(f"Omnibus test:             {friedman_path}")
    print(f"QC report:                {qc_path}")
    if scaf_metrics_path:
        print(f"Scaffold Metrics CSV:     {scaf_metrics_path}")
        print(f"Top Scaffolds Breakdown:  {top_scaf_path}")
        print(f"Scaffold Overlap Matrix:  {scaf_overlap_path}")
    print("=" * 60)

    return final_output_path


if __name__ == "__main__":
    project_folder = "/path/to/project" #uptade
    build_ultimate_statistical_pipeline(project_folder)
