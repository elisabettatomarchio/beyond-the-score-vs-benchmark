# -*- coding: utf-8 -*-
"""
Rebuilds the input files expected by align_results.py starting from the
Supporting Information spreadsheet (Beyond_the_Score_VS_results_SI_IJMS.xlsx).

The five VS_<TARGET> sheets already contain, for every screened compound, the
experimental label and the three scores used by the benchmark. This script only
splits them back into the three per-target CSV files and renames the columns to
the names align_results.py expects, so that the fixed-budget comparison can be
reproduced from the spreadsheet alone:

    python xlsx_to_inputs.py Beyond_the_Score_VS_results_SI_IJMS.xlsx ./project
    python align_results.py            # after setting project_folder = ./project

The spreadsheet is Spreadsheet S1 of the article's Supporting Information, so
this path is intended for readers who obtained it from the journal rather than
cloning the repository. A clone already contains the same input files, together
with the training and test sets required by process_pipeline.ipynb and
MCS_analysis.py, under training/ - in that case this script is not needed.
"""

import os
import sys
import pandas as pd

COLUMNS = {
    'SMILES': 'smiles',
    'Label': 'exp_active',
    'Docking CNN_VS': 'CNN_VS',
    'MCS consensus max score': 'consensus_max_score',
    'QSAR probability': 'probability',
}


def export(xlsx_path, project_dir):
    xl = pd.ExcelFile(xlsx_path)
    sheets = [s for s in xl.sheet_names if s.startswith('VS_')]
    if not sheets:
        raise ValueError(f"No VS_<TARGET> sheet found in {xlsx_path}")

    for sheet in sheets:
        target = sheet[3:]
        df = pd.read_excel(xl, sheet).rename(columns=COLUMNS)

        missing = [c for c in COLUMNS.values() if c not in df.columns]
        if missing:
            raise KeyError(f"{sheet}: missing column(s) {missing}")

        paths = {
            'docking': (os.path.join(project_dir, 'training', 'docking',
                                     f'Enrichment_{target}.csv'),
                        ['smiles', 'CNN_VS', 'exp_active']),
            'mcs': (os.path.join(project_dir, 'training', 'graph analysis',
                                 f'{target}_graph.csv'),
                    ['smiles', 'consensus_max_score']),
            'qsar': (os.path.join(project_dir, 'training', target,
                                  'qsar_pipeline_results', 'VS_QSAR_predictions.csv'),
                     ['smiles', 'probability']),
        }

        for _, (path, cols) in paths.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df[cols].to_csv(path, index=False)

        print(f"{target:8s} {len(df):5d} compounds, "
              f"{int(df['exp_active'].sum()):3d} actives -> 3 files written")

    print(f"\nInput tree ready in: {os.path.abspath(project_dir)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python xlsx_to_inputs.py <spreadsheet.xlsx> <project_folder>")
    export(sys.argv[1], sys.argv[2])
