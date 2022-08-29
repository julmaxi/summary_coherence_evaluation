from typing import List, Dict, Union
from pathlib import Path
import sys

import pandas as pd


SYSTEM_NAMES = {
    "bartscore": "BAS",
    "ccl-orig-wsj": "CCL ORI",
    "ccl-roberta-large-ours-cnndm": "CCL C/D",
    "ccl-roberta-large-ours-wsj": "CCL WSJ",
    "cnn-coherence-cnndm": "NEG C/D",
    "cnn-coherence-duc03": "NEG DUC",
    "cnn-coherence-wsj": "NEG WSJ",
    "eegrid-cnndm": "EEG C/D",
    "eegrid-wsj": "EEG WSJ",
    "entity-graph": "EGR",
    "gruen": "GRN",
    "neural-entity-graph-cnndm": "GRA C/D",
    "neural-entity-graph-duc03": "GRA DUC",
    "neural-entity-graph-wsj": "GRA WSJ",
    "sumqe": "SQE",
    "unified-cnndm": "UNF C/D",
    "unified-wsj": "UNF WSJ",
    "human": "HUM",
    "random": "RND"
}

SYSTEM_FILES = {
    v: (k + ".csv") for k, v in SYSTEM_NAMES.items()
}


def read_pred_file(path: str, predictor: str, index_cols: List[str]) -> pd.Series:
    pred_scores = pd.read_csv(path)
    pred_scores.set_index(index_cols, inplace=True)
    if predictor is None:
        if len(pred_scores.columns) > 1:
            print("Warning, no column specificed and more than one column in dataset. Using first column.", file=sys.stderr)
        
        pred_scores = pred_scores[pred_scores.columns[0]]
    else:
        pred_scores = pred_scores[predictor]
    return pred_scores


def read_gold_scores(path: Union[Path, str], metric: str) -> pd.Series:
    gold_scores = pd.read_csv(path)
    index_cols = [c for c in gold_scores.columns if c[0] != "Q"]

    gold_scores.set_index(index_cols, inplace=True)
    gold_scores = gold_scores[metric]

    if "assessor" in index_cols:
        index_cols.remove("assessor")
        gold_scores = gold_scores.groupby(index_cols).mean()
    
    return gold_scores


def read_summaries(path: Union[Path, str]) -> pd.Series:
    summaries = pd.read_csv(path)
    summaries.set_index(list(summaries.columns[:-1]), inplace=True)

    return summaries[summaries.columns[0]]


def get_measure_summeval_path(system_name: str) -> Path:
    results_path = Path("predicted_scores/summeval/")
    return results_path / SYSTEM_FILES[system_name]


def get_measure_cnn_dm_shuffle_path(system_name: str) -> Path:
    results_path = Path("predicted_scores/cnn_dm_cnn_dm_1000_sample_shuffle_test/")
    return results_path / SYSTEM_FILES[system_name]


def read_measure_cnn_dm_shuffle_results(system_name: str) -> Path:
    df = pd.read_csv(get_measure_cnn_dm_shuffle_path(system_name), index_col=["id", "label"])
    return df[df.columns[0]]
