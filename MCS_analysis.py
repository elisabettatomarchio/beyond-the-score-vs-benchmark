import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score

# ============================================================
# FILE CONFIGURATION
# ============================================================
TRAIN_FILE = "train_metadata.csv"
TEST_FILE  = "test_metadata.csv"
VS_FILE    = "vs_metadata.csv" 

OUTDIR     = "vs_MCS_results"
os.makedirs(OUTDIR, exist_ok=True)

BUTINA_RADIUS = 0.4
MIN_CLUSTER_SIZE = 2
MAX_QUERIES      = 10

# ============================================================
# CALCULATION FUNCTIONS
# ============================================================
def find_dynamic_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ind the optimal threshold by maximizing Youden's Index (J)."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    
    best_thresh = float(thresholds[best_idx])
    if best_thresh > 1.0:
        best_thresh = 1.0
    return best_thresh

def calculate_metrics_with_thresh(y_true: np.ndarray, scores: np.ndarray, thresh: float) -> tuple:
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0

    # Ranking metrics (threshold-independent)
    auc   = float(roc_auc_score(y_true, scores))
    order = np.argsort(-scores)
    sorted_true   = y_true[order]
    sorted_scores = scores[order]

    scores_list = [(float(s), int(t)) for s, t in zip(sorted_scores, sorted_true)]
    try:
        bedroc = float(CalcBEDROC(scores_list, col=1, alpha=20.0))
    except Exception:
        bedroc = 0.0

    random_prop = np.sum(sorted_true) / len(sorted_true)
    top_1 = max(1, int(len(sorted_true) * 0.01))
    top_5 = max(1, int(len(sorted_true) * 0.05))
    top_10 = max(1, int(len(sorted_true) * 0.1))

    ef1 = float((np.sum(sorted_true[:top_1]) / top_1) / random_prop) if random_prop > 0 else 0.0
    ef5 = float((np.sum(sorted_true[:top_5]) / top_5) / random_prop) if random_prop > 0 else 0.0
    ef10 = float((np.sum(sorted_true[:top_10]) / top_10) / random_prop) if random_prop > 0 else 0.0

    # Classification metrics based on the DYNAMIC THRESHOLD
    y_pred = (scores >= thresh).astype(int)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    
    true_actives = int(np.sum(y_true))
    predicted_true_actives = int(np.sum((y_pred == 1) & (y_true == 1)))

    return auc, bedroc, ef1, ef5, ef10, precision, recall, predicted_true_actives, true_actives

def mcs_similarity(q_mol: Chem.Mol, t_mol: Chem.Mol) -> float:
    q_natoms, t_natoms = q_mol.GetNumAtoms(), t_mol.GetNumAtoms()
    q_nbonds, t_nbonds = q_mol.GetNumBonds(), t_mol.GetNumBonds()

    mcs_res = rdFMCS.FindMCS(
        [q_mol, t_mol],
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        ringMatchesRingOnly=True,
        timeout=5
    )
    common_atoms = mcs_res.numAtoms
    common_bonds = mcs_res.numBonds

    if common_atoms == 0:
        return 0.0

    denom_atoms = q_natoms + t_natoms - common_atoms
    denom_bonds = q_nbonds + t_nbonds - common_bonds

    sim_atoms = common_atoms / denom_atoms if denom_atoms > 0 else 0.0
    sim_bonds = common_bonds / denom_bonds if denom_bonds > 0 else 0.0

    return 0.5 * sim_atoms + 0.5 * sim_bonds


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    # 1. Load and Cluster the Training Set
    print("[1] Caricamento train set e clustering attivi...")
    train_df = pd.read_csv(TRAIN_FILE).dropna(subset=["smiles", "label"]).drop_duplicates(subset=["smiles"])
    train_actives = train_df[train_df.label == 1].reset_index(drop=True)
    
    mols, fps = [], []
    for smi in train_actives["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    n_fps = len(fps)
    dist_matrix = []
    for i in range(1, n_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dist_matrix.extend([1.0 - x for x in sims])

    clusters = Butina.ClusterData(dist_matrix, n_fps, BUTINA_RADIUS, isDistData=True, reordering=True)
    valid_clusters = [c for c in clusters if len(c) >= MIN_CLUSTER_SIZE]
    centroid_indices = [c[0] for c in valid_clusters][:MAX_QUERIES]
    query_mols = [mols[idx] for idx in centroid_indices]

    # 2. Screen the Test Set and Compute Optimal Dynamic Thresholds
    print("\n[2] Analisi del Test Set e calcolo delle soglie dinamiche ottimali...")
    test_df = pd.read_csv(TEST_FILE).dropna(subset=["smiles", "label"]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    y_test = test_df["label"].to_numpy()
    
    test_query_cols = [f"query_{i+1}_score" for i in range(len(query_mols))]
    for col in test_query_cols:
        test_df[col] = 0.0

    for idx, row in test_df.iterrows():
        t_mol = Chem.MolFromSmiles(row["smiles"])
        if t_mol is None:
            continue
        for q_idx, q_mol in enumerate(query_mols):
            test_df.at[idx, test_query_cols[q_idx]] = mcs_similarity(q_mol, t_mol)

    test_df["consensus_max_score"] = test_df[test_query_cols].max(axis=1)

    # Compute dynamic thresholds based on the Test Set
    best_auc = -1.0
    best_query_idx = -1
    test_dynamic_thresholds = []

    for q_idx, col in enumerate(test_query_cols):
        scores = test_df[col].to_numpy()
        thresh = find_dynamic_threshold(y_test, scores)
        test_dynamic_thresholds.append(thresh)
        
        auc = float(roc_auc_score(y_test, scores))
        if auc > best_auc:
            best_auc = auc
            best_query_idx = q_idx

    consensus_test_thresh = find_dynamic_threshold(y_test, test_df["consensus_max_score"].to_numpy())
    best_query_test_thresh = test_dynamic_thresholds[best_query_idx]

    best_q_smiles = train_actives.iloc[centroid_indices[best_query_idx]]["smiles"]
    print(f" -> Best Query identificata: Query {best_query_idx+1}")
    print(f" -> Soglia Dinamica calcolata per Best Query: {best_query_test_thresh:.4f}")
    print(f" -> Soglia Dinamica calcolata per Consensus:   {consensus_test_thresh:.4f}")

    # 3. Load and Screen the Real Virtual Screening Dataset
    print("\n[3] Caricamento del database di Virtual Screening (VS)...")
    vs_df = pd.read_csv(VS_FILE).dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    has_labels = "label" in vs_df.columns
    print(f" -> Strutture da analizzare: {len(vs_df)} (Label rilevate: {has_labels})")

    vs_cols = [f"query_{i+1}_score" for i in range(len(query_mols))]
    for col in vs_cols:
        vs_df[col] = 0.0

    print("\n[4] Screening del database VS in corso...")
    for idx, row in tqdm(vs_df.iterrows(), total=len(vs_df), desc="Production VS"):
        t_mol = Chem.MolFromSmiles(row["smiles"])
        if t_mol is None:
            continue
        for q_idx, q_mol in enumerate(query_mols):
            vs_df.at[idx, vs_cols[q_idx]] = mcs_similarity(q_mol, t_mol)
            
    vs_df["best_query_score"] = vs_df[vs_cols[best_query_idx]]
    vs_df["consensus_max_score"] = vs_df[vs_cols].max(axis=1)

    # Export complete results to CSV
    csv_out = os.path.join(OUTDIR, "final_vs_scores_dynamic_thresholds.csv")
    vs_df.to_csv(csv_out, index=False)
    print(f" -> CSV dei punteggi VS salvato in: {csv_out}")

    # 4. Generate the VS Metrics Report Using Dynamic Thresholds
    if has_labels:
        print("\n[5] Metrics with dynamic thresholds...")
        y_vs = vs_df["label"].to_numpy()
        txt_vs_out = os.path.join(OUTDIR, "vs_dynamic_metrics_summary.txt")
        
        with open(txt_vs_out, "w") as f:
            f.write("=" * 75 + "\n")
            f.write("VIRTUAL SCREENING METRICS REPORT – DYNAMIC THRESHOLDS\n")
            f.write("Thresholds mathematically calibrated on the Test Set using Youden's Index (J)\n")
            f.write("=" * 75 + "\n\n")
            f.write(f"Best individual query: Query {best_query_idx+1} (SMILES: {best_q_smiles})\n\n")
            
            # SWrite metrics for individual clusters using their respective dynamic thresholds
            for q_idx, col in enumerate(vs_cols):
                scores = vs_df[col].to_numpy()
                th = test_dynamic_thresholds[q_idx]
                auc, bed, ef1, ef5, ef10, prec, rec, pred_act, tot_act = calculate_metrics_with_thresh(y_vs, scores, th)
                
                f.write(f"CLUSTER/QUERY {q_idx+1} (Soglia Adattiva = {th:.4f}):\n")
                f.write(f"  AUC={auc:.4f} | BEDROC={bed:.4f}\n")
                f.write(f"  EF1%={ef1:.4f} | EF5%={ef5:.4f} | EF10%={ef10:.4f}\n")
                f.write(f"  Precision={prec:.4f} | Recall={rec:.4f}\n")
                f.write(f"  Actives Identified: {pred_act} su {tot_act} totali\n")
                f.write("-" * 50 + "\n")
                
            # Compute the summary of the final screening strategies
            b_scores = vs_df["best_query_score"].to_numpy()
            c_scores = vs_df["consensus_max_score"].to_numpy()
            
            b_auc, b_bed, b_ef1, b_ef5, b_ef10, b_prec, b_rec, b_p_act, b_t_act = calculate_metrics_with_thresh(y_vs, b_scores, best_query_test_thresh)
            c_auc, c_bed, c_ef1, c_ef5, c_ef10, c_prec, c_rec, c_p_act, c_t_act = calculate_metrics_with_thresh(y_vs, c_scores, consensus_test_thresh)
            
            f.write("\n" + "=" * 75 + "\n")
            f.write("SUMMARY:\n")
            f.write("=" * 75 + "\n")
            f.write(f"BEST QUERY SINGOLA (Query {best_query_idx+1} | Soglia={best_query_test_thresh:.4f}) ->\n")
            f.write(f"  AUC={b_auc:.4f} | BEDROC={b_bed:.4f} | EF1%={b_ef1:.4f} | EF5%={b_ef5:.4f} | EF10%={b_ef10:.4f}\n")
            f.write(f"  Precision={b_prec:.4f} | Recall={b_rec:.4f} | Attivi Predetti={b_p_act}/{b_t_act}\n\n")
            
            f.write(f"CONSENSUS (MAX) (Soglia={consensus_test_thresh:.4f}) ->\n")
            f.write(f"  AUC={c_auc:.4f} | BEDROC={c_bed:.4f} | EF1%={c_ef1:.4f} | EF5%={c_ef5:.4f} | EF10%={c_ef10:.4f}\n")
            f.write(f"  Precision={c_prec:.4f} | Recall={c_rec:.4f} | Attivi Predetti={c_p_act}/{c_t_act}\n")
            f.write("=" * 75 + "\n")
            
        print(f" -> Report  VS saved in: {txt_vs_out}")

if __name__ == "__main__":
    main()
