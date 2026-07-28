# -*- coding: utf-8 -*-

# Output:
#  - SINGLE_TARGET_PERFORMANCE_3M.csv   (data per target/cutoff/strategy)
# - FINAL_STATISTICAL_VALIDATION_3M.csv (mean ± SD for 5 target)
#  - FRIEDMAN_OMNIBUS_3M.csv            (omnibus test across strategies, per cutoff)
#  - QC_REPORT.csv                      (summary)

import os
import glob
import pandas as pd
import numpy as np


EXPECTED_ACTIVES_PER_TARGET = 50      
KEEP_STEREOCHEMISTRY = True           
PREFILTER_MULTIPLIER_2STAGE = 2       
PREFILTER_MULTIPLIER_3STAGE_1 = 3     
PREFILTER_MULTIPLIER_3STAGE_2 = 2     

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
    print("RDKit detected: SMILES will be chemically canonicalized "
          f"(stereochemistry {'kept' if KEEP_STEREOCHEMISTRY else 'removed'}).")
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not found. Falling back to plain string cleaning (strip only).")

try:
    from scipy.stats import friedmanchisquare
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not found. The Friedman omnibus test will be skipped "
          "(install with: pip install scipy).")


def canonicalize_smiles(smiles_str):
    """Converts a SMILES string to its canonical form. Stereochemistry is kept
    by default (see FIX 2) to avoid collapsing distinct stereoisomers."""
    if not RDKIT_AVAILABLE or pd.isna(smiles_str):
        return str(smiles_str).strip()
    try:
        mol = Chem.MolFromSmiles(str(smiles_str).strip())
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=KEEP_STEREOCHEMISTRY)
    except Exception:
        pass
    return str(smiles_str).strip()


def dedupe_with_label_check(df, label_col, source_name, qc_log):
    """[FIX 3] Removes duplicate SMILES, but if a group of duplicates has
    inconsistent labels (the same canonical SMILES appearing as both active
    and decoy) it is dropped entirely and logged in the QC log, instead of
    arbitrarily keeping "the first occurrence"."""
    if label_col not in df.columns:
        # File with no label column (e.g. MCS, QSAR)
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
    """Recall/precision as percentages. The recall denominator is fixed
    (EXPECTED_ACTIVES_PER_TARGET, validated by QC), not recomputed ad hoc for
    each strategy."""
    true_positives = int(np.sum(selected_mask & (true_labels == 1)))
    total_selected = int(np.sum(selected_mask))
    recall = (true_positives / actives_denominator) * 100 if actives_denominator > 0 else 0.0
    precision = (true_positives / total_selected) * 100 if total_selected > 0 else 0.0
    return recall, precision, true_positives, total_selected



def top_b_by_score(score, B, ascending=False):
    """Returns the (positional) indices of the best B elements of `score`.
    Ties are broken deterministically by original array order (first
    occurrence wins), for both ascending and descending selection."""
    order = np.argsort(score if ascending else -score, kind='mergesort')
    return order[:B]


def rank_min(score):
    """Rank 1 = best, ties handled with the 'min' method (same behavior as
    pandas .rank(method='min'), rewritten to work on numpy arrays)."""
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
    """ Sequential funnel: at each stage, the library (or the
    survivors from the previous stage) is PRE-FILTERED based on the score of
    the current stage, then the next stage RECOMPUTES its own ranking ONLY on
    the survivors (not on the entire library).

    scores_stage_order : list of score arrays (one per stage, all indexed
                          0..N-1 over the full library)
    prefilter_sizes     : pre-filter size for every stage EXCEPT the last
                          (the last stage selects exactly B compounds)
    """
    survivors = np.arange(N)
    for stage_idx, score in enumerate(scores_stage_order[:-1]):
        P = min(len(survivors), max(B, prefilter_sizes[stage_idx]))
        stage_scores = score[survivors]
        top_local = top_b_by_score(stage_scores, P, ascending=False)
        survivors = survivors[top_local]
    # Last stage: final re-ranking on the survivors, take exactly B
    last_score = scores_stage_order[-1][survivors]
    final_local = top_b_by_score(last_score, min(B, len(survivors)), ascending=False)
    return survivors[final_local]


def run_friedman_omnibus(df_all, output_dir, cutoff_labels):
    """Omnibus Friedman rank-sum test across the screening strategies, run
    separately at every budget.

    Blocks     = benchmark targets
    Treatments = screening strategies
    Response   = TP, the number of recovered actives (the fixed budget makes
                 recall and precision monotonic transformations of TP, so the
                 test is invariant to which of the three is used)

    The test answers one question only: are the strategies all equivalent?
    It is NOT a licence for pairwise claims. With a handful of targets the
    chi-square approximation is asymptotic, and no post-hoc comparison is
    attempted: the smallest two-sided p value reachable by a Wilcoxon
    signed-rank test over n paired targets is 2^-(n-1), i.e. 0.0625 for n = 5,
    so no pairwise contrast can ever reach alpha = 0.05 on this panel.
    """
    if not SCIPY_AVAILABLE:
        print("Friedman omnibus test skipped: SciPy is not installed.\n")
        return None

    rows = []
    print("Friedman omnibus test (blocks = targets, response = recovered actives)")

    for c_lbl in cutoff_labels:
        pivot = df_all[df_all['Cutoff'] == c_lbl].pivot_table(
            index='Target', columns='Strategy', values='TP')
        # A target missing any strategy would silently unbalance the design
        n_incomplete = int(pivot.isna().any(axis=1).sum())
        pivot = pivot.dropna(axis=0, how='any')
        n_blocks, n_strategies = pivot.shape

        if n_blocks < 3 or n_strategies < 3:
            print(f"   {c_lbl}: skipped (targets={n_blocks}, strategies={n_strategies}; "
                  "the test needs at least 3 of each).")
            continue

        chi2, p_value = friedmanchisquare(*[pivot[s].values for s in pivot.columns])

        # rank 1 = largest TP count, tied strategies share the same rank
        mean_rank = pivot.rank(axis=1, ascending=False, method='min').mean(axis=0)
        best_strategy = mean_rank.idxmin()

        note = ""
        if n_incomplete:
            note = f"{n_incomplete} target(s) dropped for incomplete strategy coverage; "
        if n_blocks < 10:
            note += (f"asymptotic approximation with only {n_blocks} blocks - "
                     "read as a global heterogeneity check, not as pairwise evidence")

        rows.append({
            'Cutoff': c_lbl,
            'N_targets': n_blocks,
            'N_strategies': n_strategies,
            'df': n_strategies - 1,
            'Chi2': round(float(chi2), 3),
            'p_value': float(p_value),
            'Best_strategy_by_mean_rank': best_strategy,
            'Best_mean_rank': round(float(mean_rank.min()), 3),
            'Note': note.strip()
        })

        print(f"   {c_lbl}: chi2({n_strategies - 1}) = {chi2:.1f}, p = {p_value:.2e} "
              f"| best mean rank: {best_strategy} ({mean_rank.min():.2f})")

    if not rows:
        print("   No cutoff produced a usable design; nothing written.\n")
        return None

    friedman_path = os.path.join(output_dir, "FRIEDMAN_OMNIBUS_STRATEGY_COMPARISON.csv")
    pd.DataFrame(rows).to_csv(friedman_path, index=False)
    print("   Reminder: no post-hoc pairwise test is reported - with five targets "
          "none could reach significance.\n")
    return friedman_path


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

    print("Starting data merge and metric calculation (20 strategies, fixed budget)...\n")

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
            elif len(possible_cols) > 1:
                raise KeyError(
                    f"Ambiguous label column in {os.path.basename(dock_path)}: "
                    f"candidates found {possible_cols}. Specify explicitly which one to use."
                )
            else:
                raise KeyError(f"Missing the true-actives column in {os.path.basename(dock_path)}")

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

        print(f"   Rows before merge -> Docking: {len(df_dock_clean)} | "
              f"MCS: {len(df_mcs_clean)} | QSAR: {len(df_qsar_clean)}")

        # 3. MERGE (inner join on SMILES) with explicit diagnostics 
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

        print(f"   Merge completed: {N_total} aligned molecules "
              f"({molecules_lost} lost relative to the original docking file).")
        print(f"   Actives in the final merge: {n_actives_merged} "
              f"(expected: {EXPECTED_ACTIVES_PER_TARGET}, lost in merge: {actives_lost}).")

        if n_actives_merged != EXPECTED_ACTIVES_PER_TARGET:
            print(f"   *** QC FAILED for {target_name}: {n_actives_merged} actives "
                  f"instead of {EXPECTED_ACTIVES_PER_TARGET}. See QC_REPORT.csv. ***")
        qc_log.append({
            'source': target_name, 'issue': 'merge_summary',
            'n_dock_raw': n_dock_raw, 'n_actives_dock_raw': n_actives_dock_raw,
            'N_total_merged': N_total, 'n_actives_merged': n_actives_merged,
            'molecules_lost': molecules_lost, 'actives_lost': actives_lost,
            'qc_pass_50_actives': n_actives_merged == EXPECTED_ACTIVES_PER_TARGET
        })

        df_merged = df_merged.reset_index(drop=True)
        labels = df_merged[target_label_col].values.astype(int)
        actives_denominator = n_actives_merged  # real, verified recall denominator

        score_qsar = df_merged['Score_ML'].values.astype(float)
        score_mcs = df_merged['Score_MCS'].values.astype(float)
        score_dock = df_merged['CNN_VS'].values.astype(float)

        rank_qsar = rank_min(score_qsar)
        rank_mcs = rank_min(score_mcs)
        rank_dock = rank_min(score_dock)

        # 4. Metrics for the 3 cutoffs x 20 strategies, all at fixed budget B 
        for c_val, c_lbl in zip(cutoffs, cutoff_labels):
            B = max(1, int(N_total * c_val))  # k = floor(f*N)

            selections = {}

            # --- Single methods ---
            selections['Single QSAR'] = top_b_by_score(score_qsar, B)
            selections['Single MCS'] = top_b_by_score(score_mcs, B)
            selections['Single Docking'] = top_b_by_score(score_dock, B)

            # --- Rank-fusion strategies -------------------------------------------------
            # Because every strategy must return exactly B compounds, these are rank-fusion operators that emulate an OR-like
            # or AND-like combination while keeping the selected-set size fixed.
            #   - "Best-rank fusion" (OR-like): combined score = MINIMUM rank across the
            #     constituent methods (a compound only needs to rank well in ONE method).
            #   - "Worst-rank fusion" (AND-like): combined score = MAXIMUM rank across the
            #     constituent methods (a compound must rank well in ALL methods).
            # ------------------------------------------------------------------------------
            selections['Best-Rank Fusion (QSAR-MCS)'] = top_b_by_score(
                np.minimum(rank_qsar, rank_mcs), B, ascending=True)
            selections['Best-Rank Fusion (3-Method)'] = top_b_by_score(
                np.minimum(np.minimum(rank_qsar, rank_mcs), rank_dock), B, ascending=True)

            selections['Worst-Rank Fusion (QSAR-MCS)'] = top_b_by_score(
                np.maximum(rank_qsar, rank_mcs), B, ascending=True)
            selections['Worst-Rank Fusion (3-Method)'] = top_b_by_score(
                np.maximum(np.maximum(rank_qsar, rank_mcs), rank_dock), B, ascending=True)

            # --- Consensus: average of ranks ---
            mean_rank = (rank_qsar + rank_mcs + rank_dock) / 3.0
            selections['Consensus (Mean Rank 3M)'] = top_b_by_score(mean_rank, B, ascending=True)

            # --- Two-stage sequential funnels: true re-ranking on survivors  ---
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

            # --- Three-stage sequential funnels: true cascading re-ranking  ---
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

            # 5. Metric calculation
            for strat_name, idx in selections.items():
                mask = np.zeros(N_total, dtype=bool)
                mask[idx] = True
                rec, prec, tp, n_sel_actual = calculate_metrics(mask, labels, actives_denominator)
                all_target_results.append({
                    'Target': target_name,
                    'Cutoff': c_lbl,
                    'Strategy': strat_name,
                    'Budget_B': B,
                    'Selected_N': n_sel_actual,   # for transparency: should equal B
                    'TP': tp,
                    'Recall': rec,
                    'Precision': prec
                })
        print(f"Metrics calculated for {target_name}\n")

    if not all_target_results:
        print("Error: no data collected. Check the file/path naming.")
        return None

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

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Raw data:           {single_target_path}")
    print(f"Summary statistics: {final_output_path}")
    if friedman_path:
        print(f"Omnibus test:       {friedman_path}")
    print(f"QC report:          {qc_path}  <-- check this first")
    print("=" * 60)

    failed_qc = [q for q in qc_log if q.get('issue') == 'merge_summary' and not q['qc_pass_50_actives']]
    if failed_qc:
        print(f"WARNING: {len(failed_qc)} target(s) do not have exactly "
              f"{EXPECTED_ACTIVES_PER_TARGET} actives after the merge. See QC_REPORT for details.")

    return final_output_path


if __name__ == "__main__":
    project_folder = "/Users/your_path"  # update with your folder
    build_ultimate_statistical_pipeline(project_folder)
