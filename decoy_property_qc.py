# -*- coding: utf-8 -*-
"""
Decoy quality-control analysis: active vs. decoy comparison per target,
scaffold diversity analysis (50 VS actives vs training actives),
and explicit comparison against training-set true inactives and actives.

Outputs (written to OUTDIR):
  - compound_properties.csv        raw per-compound descriptors
  - smd_summary.csv                SMD per property per target (+ pooled across targets)
  - scaffold_diversity_summary.csv scaffold diversity ratio & active/decoy overlap per target
  - vs_50_scaffold_analysis.csv     Bemis-Murcko scaffold diversity & novelty analysis (50 VS actives vs Train actives)
  - nn_similarity_vs_train.csv      per-molecule max Tanimoto similarities across comparison types
  - nn_similarity_bins_summary.csv  % of molecules per target in each similarity bin per comparison
  - FIG_property_violins.png        violin plots: VS-Active vs VS-Decoy
  - FIG_smd_heatmap.png             SMD heatmap (targets x properties)
  - FIG_scaffold_diversity.png      scaffold diversity ratio bar chart
  - FIG_scaffold_top50_novelty.png    stacked bar chart: novel vs overlapping scaffolds in Top 50 VS actives
  - FIG_nn_similarity.png            nearest-neighbor similarity distributions per comparison
"""
import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')  # silence RDKit parsing warnings

# ============================================================
# CONFIGURATION
# ============================================================
TARGETS = ['AChE', 'EGFR', 'hERG', 'HIV1-P', 'PPARgamma']

VS_XLSX_PATH = None  # Set to path string if using Excel workbook, else None for CSVs
VS_SHEET_TEMPLATE = "VS_{target}"

BASE_DIR = "/path/to/project"

VS_CSV_PATHS = {
    'AChE': os.path.join(
        BASE_DIR,
        'training/AChE/qsar_pipeline_results/VS_QSAR_predictions.csv'
    ),
    'EGFR': os.path.join(
        BASE_DIR,
        'training/EGFR/qsar_pipeline_results/VS_QSAR_predictions.csv'
    ),
    'hERG': os.path.join(
        BASE_DIR,
        'training/hERG/qsar_pipeline_results/VS_QSAR_predictions.csv'
    ),
    'HIV1-P': os.path.join(
        BASE_DIR,
        'training/HIV1-P/qsar_pipeline_results/VS_QSAR_predictions.csv'
    ),
    'PPARgamma': os.path.join(
        BASE_DIR,
        'training/PPARgamma/qsar_pipeline_results/VS_QSAR_predictions.csv'
    ),
}
VS_SMILES_COL = "SMILES"
VS_LABEL_COL = "Label"

TRAIN_CSV_PATHS = {
    'AChE': os.path.join(
        BASE_DIR,
        'training/AChE/qsar_pipeline_results/train_metadata.csv'
    ),
    'EGFR': os.path.join(
        BASE_DIR,
        'training/EGFR/qsar_pipeline_results/train_metadata.csv'
    ),
    'hERG': os.path.join(
        BASE_DIR,
        'training/hERG/qsar_pipeline_results/train_metadata.csv'
    ),
    'HIV1-P': os.path.join(
        BASE_DIR,
        'training/HIV1-P/qsar_pipeline_results/train_metadata.csv'
    ),
    'PPARgamma': os.path.join(
        BASE_DIR,
        'training/PPARgamma/qsar_pipeline_results/train_metadata.csv'
    ),
}
TRAIN_SMILES_COL = "smiles"
TRAIN_LABEL_COL = "label"

OUTDIR = "decoy_qc_output"

# Fingerprint settings
ECFP_RADIUS = 2   # ECFP4
ECFP_NBITS = 2048

SIMILARITY_BINS = [0.0, 0.3, 0.5, 0.7, 1.0001]
SIMILARITY_BIN_LABELS = ['<0.3', '0.3-0.5', '0.5-0.7', '>=0.7']

# ============================================================
# DESCRIPTOR / SCAFFOLD / FINGERPRINT HELPERS
# ============================================================

PROPERTY_FUNCS = {
    'MW': Descriptors.MolWt,
    'cLogP': Crippen.MolLogP,
    'TPSA': Descriptors.TPSA,
    'HBD': Descriptors.NumHDonors,
    'HBA': Descriptors.NumHAcceptors,
    'RotatableBonds': Descriptors.NumRotatableBonds,
    'FormalCharge': Chem.GetFormalCharge,
    'FractionCSP3': rdMolDescriptors.CalcFractionCSP3,
}


def aromatic_proportion(mol):
    """Fraction of heavy atoms that are aromatic."""
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return 0.0
    n_aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    return n_aromatic / heavy


def compute_properties(mol):
    props = {name: func(mol) for name, func in PROPERTY_FUNCS.items()}
    props['AromaticProportion'] = aromatic_proportion(mol)
    return props


def murcko_scaffold_smiles(mol):
    """Computes Bemis-Murcko Scaffold SMILES. Returns None for acyclic compounds."""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return None
        smiles = Chem.MolToSmiles(scaffold)
        return smiles if smiles != "" else None
    except Exception:
        return None


def ecfp4(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, ECFP_RADIUS, nBits=ECFP_NBITS)


def smd(a, b):
    """Standardized mean difference (Cohen's d) using weighted pooled SD."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan
    
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_sd = np.sqrt(pooled_var)
    
    if pooled_sd == 0:
        return 0.0 if a.mean() == b.mean() else np.nan
    return (a.mean() - b.mean()) / pooled_sd


# ============================================================
# DATA LOADING
# ============================================================

def load_vs_set(target):
    """Loads VS set with explicit group labels: 'VS-Active' (1) and 'VS-Decoy' (0)."""
    if VS_XLSX_PATH and os.path.exists(VS_XLSX_PATH):
        sheet = VS_SHEET_TEMPLATE.format(target=target)
        df = pd.read_excel(VS_XLSX_PATH, sheet_name=sheet)
    elif target in VS_CSV_PATHS and os.path.exists(VS_CSV_PATHS[target]):
        df = pd.read_csv(VS_CSV_PATHS[target])
    else:
        raise FileNotFoundError(f"No valid VS-set input file or sheet found for target '{target}'.")
        
    df = df.rename(columns={VS_SMILES_COL: 'smiles', VS_LABEL_COL: 'label'})
    df = df[['smiles', 'label']].dropna(subset=['smiles', 'label']).copy()
    df['label'] = df['label'].astype(int)
    df['group'] = df['label'].map({1: 'VS-Active', 0: 'VS-Decoy'})
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    
    n_failed = df['mol'].isna().sum()
    if n_failed:
        print(f"  [{target}] WARNING: {n_failed} VS-set SMILES failed to parse and were dropped.")
    return df.dropna(subset=['mol']).reset_index(drop=True)


def load_train_set(target):
    """Loads training set with explicit group labels: 'Train-Active' (1) and 'Train-Inactive' (0)."""
    path = TRAIN_CSV_PATHS.get(target)
    if not path or not os.path.exists(path):
        print(f"  [{target}] WARNING: training-set file not found ({path}); "
              f"training-set comparisons will be skipped for this target.")
        return None
        
    df = pd.read_csv(path)
    df = df.rename(columns={TRAIN_SMILES_COL: 'smiles', TRAIN_LABEL_COL: 'label'})
    df = df.dropna(subset=['smiles', 'label']).copy()
    df['label'] = df['label'].astype(int)
    df['group'] = df['label'].map({1: 'Train-Active', 0: 'Train-Inactive'})
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    
    n_failed = df['mol'].isna().sum()
    if n_failed:
        print(f"  [{target}] WARNING: {n_failed} training SMILES failed to parse and were dropped.")
    return df.dropna(subset=['mol']).reset_index(drop=True)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    all_props_rows = []
    all_scaffold_rows = []
    all_top50_scaffold_rows = []
    all_nn_rows = []
    umap_rows = []

    for target in TARGETS:
        print(f"--- Processing {target} ---")
        vs_df = load_vs_set(target)
        train_df = load_train_set(target)

        # ---- Collect molecules for global UMAP ---------------------------------------------
        for _, r in vs_df.iterrows():
            umap_rows.append({
                "Target": target,
                "Group": r["group"],
                "fp": ecfp4(r["mol"])
            })

        if train_df is not None:
            for _, r in train_df.iterrows():
                umap_rows.append({
                    "Target": target,
                    "Group": r["group"],
                    "fp": ecfp4(r["mol"])
                })

        # ---- 1. Physicochemical properties (VS Actives vs VS Decoys) -----------------------
        for _, row in vs_df.iterrows():
            props = compute_properties(row['mol'])
            props.update({
                'Target': target,
                'Label': row['group'],
                'smiles': row['smiles']
            })
            all_props_rows.append(props)

        # ---- 2a. Global Scaffold diversity (VS Actives vs VS Decoys) ----------------------
        vs_df['scaffold'] = vs_df['mol'].apply(murcko_scaffold_smiles)
        for label_group in ['VS-Active', 'VS-Decoy']:
            grp = vs_df[vs_df['group'] == label_group]
            n = len(grp)
            n_scaffolds = grp['scaffold'].dropna().nunique()
            all_scaffold_rows.append({
                'Target': target, 'Group': label_group, 'N_compounds': n,
                'N_unique_scaffolds': n_scaffolds,
                'Scaffold_diversity_ratio': n_scaffolds / n if n > 0 else np.nan
            })
            
        active_scaffolds = set(vs_df[vs_df['group'] == 'VS-Active']['scaffold'].dropna())
        decoy_scaffolds = set(vs_df[vs_df['group'] == 'VS-Decoy']['scaffold'].dropna())
        union = active_scaffolds | decoy_scaffolds
        jaccard = len(active_scaffolds & decoy_scaffolds) / len(union) if union else np.nan
        all_scaffold_rows.append({
            'Target': target, 'Group': 'Active-Decoy overlap (Jaccard)',
            'N_compounds': np.nan, 'N_unique_scaffolds': np.nan,
            'Scaffold_diversity_ratio': jaccard
        })

        # ---- 2b. Bemis-Murcko Scaffold Analysis: Top 50 VS Actives vs Train Actives --------
        vs_actives = vs_df[vs_df['group'] == 'VS-Active']
        vs_actives_top50 = vs_actives.head(50).copy()
        
        vs_top50_scaffolds = [murcko_scaffold_smiles(m) for m in vs_actives_top50['mol']]
        vs_top50_scaffolds = [s for s in vs_top50_scaffolds if s is not None]
        
        unique_vs_scaffolds = set(vs_top50_scaffolds)
        n_vs_compounds = len(vs_actives_top50)
        n_unique_vs_scaffolds = len(unique_vs_scaffolds)
        scr = n_unique_vs_scaffolds / n_vs_compounds if n_vs_compounds > 0 else np.nan
        
        # Compound distribution per scaffold
        scaffold_counts = Counter(vs_top50_scaffolds)
        singletons = sum(1 for count in scaffold_counts.values() if count == 1)
        max_per_scaffold = max(scaffold_counts.values()) if scaffold_counts else 0
        avg_per_scaffold = np.mean(list(scaffold_counts.values())) if scaffold_counts else 0.0
        
        # Training set actives scaffolds
        if train_df is not None and len(train_df) > 0:
            train_actives = train_df[train_df['group'] == 'Train-Active']
            train_scaffolds = set([murcko_scaffold_smiles(m) for m in train_actives['mol']])
            train_scaffolds.discard(None)
        else:
            train_scaffolds = set()
            
        overlapping_scaffolds = unique_vs_scaffolds.intersection(train_scaffolds)
        novel_scaffolds = unique_vs_scaffolds - train_scaffolds
        
        n_novel = len(novel_scaffolds)
        perc_novel = (n_novel / n_unique_vs_scaffolds * 100) if n_unique_vs_scaffolds > 0 else 0.0
        
        all_top50_scaffold_rows.append({
            'Target': target,
            'VS_Actives_Evaluated': n_vs_compounds,
            'Unique_Scaffolds_Top50': n_unique_vs_scaffolds,
            'Scaffold_Compound_Ratio': round(scr, 4),
            'Singletons_Count': singletons,
            'Max_Compounds_Per_Scaffold': max_per_scaffold,
            'Avg_Compounds_Per_Scaffold': round(avg_per_scaffold, 2),
            'Train_Active_Scaffolds': len(train_scaffolds),
            'Overlapping_Scaffolds': len(overlapping_scaffolds),
            'Novel_VS_Scaffolds': n_novel,
            'Perc_Novel_VS_Scaffolds': round(perc_novel, 2)
        })

        # ---- 3. Nearest-neighbor ECFP4 similarity comparisons ------------------------------
        if train_df is not None and len(train_df) > 0:
            train_actives = train_df[train_df['group'] == 'Train-Active']
            train_inactives = train_df[train_df['group'] == 'Train-Inactive']
            
            fps_train_active = [ecfp4(m) for m in train_actives['mol']] if len(train_actives) > 0 else []
            fps_train_inactive = [ecfp4(m) for m in train_inactives['mol']] if len(train_inactives) > 0 else []

            vs_actives = vs_df[vs_df['group'] == 'VS-Active']
            vs_decoys = vs_df[vs_df['group'] == 'VS-Decoy']

            # Helper for similarity calculation
            def compute_max_sims(source_df, target_fps, comparison_label):
                if not target_fps:
                    return
                for _, r in source_df.iterrows():
                    fp = ecfp4(r['mol'])
                    sims = DataStructs.BulkTanimotoSimilarity(fp, target_fps)
                    all_nn_rows.append({
                        'Target': target,
                        'Query_Group': r['group'],
                        'Comparison': comparison_label,
                        'smiles': r['smiles'],
                        'max_tanimoto': max(sims) if sims else np.nan
                    })

            # Comparison 1: VS-Active vs Train-Active (Analogue bias)
            compute_max_sims(vs_actives, fps_train_active, 'VS-Active vs Train-Active')
            
            # Comparison 2: VS-Decoy vs Train-Active (Check if decoys mimic active space)
            compute_max_sims(vs_decoys, fps_train_active, 'VS-Decoy vs Train-Active')

            # Comparison 3: VS-Decoy vs Train-Inactive (Check decoy overlap with True Inactives)
            compute_max_sims(vs_decoys, fps_train_inactive, 'VS-Decoy vs Train-Inactive')

        print(f"  Done: {len(vs_df)} VS compounds "
              f"({(vs_df['group'] == 'VS-Active').sum()} actives, {(vs_df['group'] == 'VS-Decoy').sum()} decoys)")

    # ---- Assemble dataframes ---------------------------------------------------------------
    props_df = pd.DataFrame(all_props_rows)
    scaffold_df = pd.DataFrame(all_scaffold_rows)
    top50_scaffold_df = pd.DataFrame(all_top50_scaffold_rows)
    nn_df = pd.DataFrame(all_nn_rows, columns=['Target', 'Query_Group', 'Comparison', 'smiles', 'max_tanimoto'])

    # ============================================================
    # UMAP projection of chemical space
    # ============================================================

    if len(umap_rows):
        print("Building global UMAP...")
        
        X = np.asarray([
            np.asarray(list(fp.ToBitString()), dtype=np.uint8)
            for fp in [r["fp"] for r in umap_rows]
        ])

        reducer = umap.UMAP(
            n_neighbors=25,
            min_dist=0.3,
            metric="jaccard",
            random_state=42
        )

        embedding = reducer.fit_transform(X)

        umap_df = pd.DataFrame({
            "UMAP1": embedding[:,0],
            "UMAP2": embedding[:,1],
            "Target":[r["Target"] for r in umap_rows],
            "Group":[r["Group"] for r in umap_rows]
        })

        umap_df.to_csv(
            os.path.join(OUTDIR,"umap_coordinates.csv"),
            index=False
        )

    props_df.to_csv(os.path.join(OUTDIR, "compound_properties.csv"), index=False)
    scaffold_df.to_csv(os.path.join(OUTDIR, "scaffold_diversity_summary.csv"), index=False)
    top50_scaffold_df.to_csv(os.path.join(OUTDIR, "vs_50_scaffold_analysis.csv"), index=False)

    # ---- SMD summary -------------------------------------------------------------------------
    property_cols = list(PROPERTY_FUNCS.keys()) + ['AromaticProportion']
    smd_rows = []
    for target in TARGETS:
        sub = props_df[props_df['Target'] == target]
        active = sub[sub['Label'] == 'VS-Active']
        decoy = sub[sub['Label'] == 'VS-Decoy']
        row = {'Target': target}
        for prop in property_cols:
            row[prop] = smd(active[prop], decoy[prop])
        smd_rows.append(row)
        
    active_all = props_df[props_df['Label'] == 'VS-Active']
    decoy_all = props_df[props_df['Label'] == 'VS-Decoy']
    pooled_row = {'Target': 'ALL (pooled)'}
    for prop in property_cols:
        pooled_row[prop] = smd(active_all[prop], decoy_all[prop])
    smd_rows.append(pooled_row)
    
    smd_df = pd.DataFrame(smd_rows).set_index('Target')
    smd_df.to_csv(os.path.join(OUTDIR, "smd_summary.csv"))

    # ---- Nearest-neighbor similarity outputs --------------------------------------------------
    if not nn_df.empty:
        nn_df.to_csv(os.path.join(OUTDIR, "nn_similarity_vs_train.csv"), index=False)
        nn_df['bin'] = pd.cut(nn_df['max_tanimoto'], bins=SIMILARITY_BINS,
                              labels=SIMILARITY_BIN_LABELS, right=False)
        
        counts = nn_df.groupby(['Target', 'Comparison', 'bin'], observed=True).size().rename('count').reset_index()
        totals = counts.groupby(['Target', 'Comparison'])['count'].transform('sum')
        counts['percent'] = 100 * counts['count'] / totals
        counts.to_csv(os.path.join(OUTDIR, "nn_similarity_bins_summary.csv"), index=False)

    # ============================================================
    # Summary statistics of nearest-neighbour similarities
    # ============================================================

    if not nn_df.empty:
        summary = (
            nn_df
            .groupby(["Comparison"])
            .agg(
                N=("max_tanimoto","count"),
                Mean=("max_tanimoto","mean"),
                Median=("max_tanimoto","median"),
                SD=("max_tanimoto","std"),
                Min=("max_tanimoto","min"),
                Q1=("max_tanimoto",lambda x:x.quantile(0.25)),
                Q3=("max_tanimoto",lambda x:x.quantile(0.75)),
                Max=("max_tanimoto","max")
            )
            .reset_index()
        )

        summary.to_csv(
            os.path.join(
                OUTDIR,
                "nn_similarity_summary.csv"
            ),
            index=False
        )

    # ============================================================
    # FIGURES
    # ============================================================
    sns.set_style('whitegrid')

    # --- Violin plots: Physicochemical properties ---
    n_props = len(property_cols)
    ncols = 3
    nrows = int(np.ceil(n_props / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    
    for i, prop in enumerate(property_cols):
        ax = axes[i]
        sns.violinplot(data=props_df, x='Target', y=prop, hue='Label', split=True,
                        inner='quartile', ax=ax, cut=0, linewidth=0.8)
        ax.set_title(prop, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)
        
        leg = ax.get_legend()
        if leg is not None:
            if i == 0:
                sns.move_legend(ax, "upper right", title=None, fontsize=8)
            else:
                leg.remove()

    for j in range(n_props, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle('Physicochemical Property Distributions: VS-Active vs. VS-Decoy',
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "FIG_property_violins.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- SMD heatmap ---
    plt.figure(figsize=(1.1 * len(property_cols) + 2, 0.6 * len(smd_df) + 2))
    sns.heatmap(smd_df[property_cols].astype(float), annot=True, fmt=".2f", cmap='RdBu_r',
                center=0, vmin=-1.5, vmax=1.5, linewidths=0.5,
                cbar_kws={'label': "Standardized mean difference (Cohen's d)"})
    plt.title("Active-Decoy Separation per Property (SMD)", fontweight='bold')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "FIG_smd_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Scaffold diversity bar chart ---
    div_plot_df = scaffold_df[scaffold_df['Group'].isin(['VS-Active', 'VS-Decoy'])]
    plt.figure(figsize=(8, 5))
    sns.barplot(data=div_plot_df, x='Target', y='Scaffold_diversity_ratio', hue='Group')
    plt.ylabel('Unique scaffolds / compounds')
    plt.title('Scaffold Diversity: VS-Active vs. VS-Decoy', fontweight='bold')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "FIG_scaffold_diversity.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Top 50 VS Actives Scaffold Novelty stacked bar chart ---
    if not top50_scaffold_df.empty:
        plt.figure(figsize=(8, 5))
        targets_list = top50_scaffold_df['Target'].tolist()
        overlapping = top50_scaffold_df['Overlapping_Scaffolds'].values
        novel = top50_scaffold_df['Novel_VS_Scaffolds'].values

        p1 = plt.bar(targets_list, overlapping, label='Overlapping with Train Actives', color='#4C72B0')
        p2 = plt.bar(targets_list, novel, bottom=overlapping, label='Novel Scaffolds', color='#55A868')

        for i, total in enumerate(top50_scaffold_df['Unique_Scaffolds_Top50']):
            pct = top50_scaffold_df.loc[i, 'Perc_Novel_VS_Scaffolds']
            plt.text(i, total + 0.5, f"{total} ({pct}% novel)", ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.ylabel('Number of Unique Scaffolds (50 VS Actives)')
        plt.title('Top 50 VS Actives: Bemis-Murcko Scaffold Novelty vs. Training Set', fontweight='bold')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "FIG_scaffold_top50_novelty.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # --- Nearest-neighbor similarity distributions ---
    if not nn_df.empty and nn_df['max_tanimoto'].notna().any():
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=nn_df, x='Target', y='max_tanimoto', hue='Comparison',
                        cut=0, inner='quartile', linewidth=0.8)
        plt.axhline(0.7, color='red', linestyle='--', linewidth=1,
                    label='0.7 (close-analogue threshold)')
        plt.ylabel('Max ECFP4 Tanimoto Similarity')
        plt.title('Nearest-Neighbor Structural Similarity Distributions', fontweight='bold')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "FIG_nn_similarity.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # ============================================================
    # UMAP figure
    # ============================================================

    if 'umap_df' in locals():
        plt.figure(figsize=(8,7))

        palette = {
            "Train-Active":"tab:blue",
            "Train-Inactive":"tab:gray",
            "VS-Active":"tab:red",
            "VS-Decoy":"tab:green"
        }

        for g in palette:
            sub = umap_df[umap_df.Group==g]
            plt.scatter(
                sub.UMAP1,
                sub.UMAP2,
                s=12,
                alpha=0.55,
                label=g
            )

        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")

        plt.title(
            "Chemical space projection (ECFP4 fingerprints)",
            fontweight="bold"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(OUTDIR,"FIG_UMAP_training_vs_vs.png"),
            dpi=300
        )
        plt.close()

    print("\nAll outputs written to:", os.path.abspath(OUTDIR))


if __name__ == "__main__":
    main()
