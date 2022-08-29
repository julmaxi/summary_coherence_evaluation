import enum
from typing import Dict, List
from collections import OrderedDict
from wsgiref.handlers import format_date_time
from pathlib import Path
import tqdm


from .evaluate import MetricResult, METRICS, compute_metric
from .reader import read_pred_file, read_gold_scores, get_measure_summeval_path


METRIC_ORDER = ["intra", "intrad", "doc", "sys", "pair"]

def create_row(results: Dict[str, MetricResult]) -> List[str]:
    cells = []
    for metric in METRIC_ORDER:
        metric_result = results[metric]
        
        if metric_result.ci_95_lower_bound is not None:
            cells.append(f"{metric_result.score:+.2f} ({metric_result.ci_95_lower_bound:+.2f} {metric_result.ci_95_upper_bound:+.2f})")
        else:
            cells.append(f"{metric_result.score:+.2f}")
    
    return cells


def create_table(results: Dict[str, Dict[str, MetricResult]]) -> List[List[str]]:
    rows = [["Sum"] + list(METRIC_ORDER)]

    for measure, result in results.items():
        row = [measure]
        row.extend(create_row(result))
        rows.append(row)

    return rows


def format_table_print(rows: List[List[str]]) -> str:
    max_cell_widths = [0 for _ in range(len(rows[0]))]
    for row in rows:
        for col_idx, cell in enumerate(row):
            max_cell_widths[col_idx] = max(len(cell), max_cell_widths[col_idx])
    
    out_rows = []
    for row in rows:
        justified_cells = []
        for col_idx, cell in enumerate(row):
            if col_idx == 0:
                cell = cell.ljust(max_cell_widths[col_idx])
            else:
                cell = cell.rjust(max_cell_widths[col_idx])
            
            justified_cells.append(cell)
        out_rows.append("\t".join(justified_cells))
    
    return "\n".join(out_rows)


ORDER = [
    "HUM",
    "RND",
    "EGR",
    "EEG C/D",
    "EEG WSJ",
    "NEG C/D",
    "NEG DUC",
    "NEG WSJ",
    "UNF C/D",
    "UNF WSJ",
    "GRA DUC",
    "GRA C/D",
    "GRA WSJ",
    "CCL C/D",
    "CCL WSJ",
    "BAS",
    "GRN",
    "SQE"
]


def main():
    gold_scores = read_gold_scores("data/sumeval_full_scores.csv", "Qcoherence")
    n_resample = 1000

    measure_results = OrderedDict()

    for system_name in tqdm.tqdm(ORDER):
        system_pred_path = get_measure_summeval_path(system_name)
        pred_scores = read_pred_file(system_pred_path, None, list(gold_scores.index.names))
    
        metric_results = {}
        for metric, func in METRICS:
            result = compute_metric(func, gold_scores, pred_scores, n_resample=n_resample)
            metric_results[metric] = result
        measure_results[system_name] = metric_results
    
    table = create_table(measure_results)
    table_str = format_table_print(table)
    print(table_str)
    

if __name__ == "__main__":
    main()
        
    

