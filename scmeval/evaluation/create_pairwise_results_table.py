from pathlib import Path
from .create_results_table import SYSTEM_FILES
from .evaluate_pairwise import read_pairwise_file, compute_pairwise_accuracy

ORDER = [
    ("EGR", "0.889"),
    ("EEG", "0.840", ("EEG C/D", "EEG WSJ")),
    ("NEG", "0.855",  ("NEG C/D", "NEG WSJ", "NEG DUC")),
    ("GRA",  "0.924", ("GRA C/D", "GRA WSJ", "GRA DUC")),
    ("UNF",  "0.93", ("UNF C/D", "UNF WSJ")),
    ("CCL",  "0.97", ("CCL C/D", "CCL WSJ")),
    ("BAS",  "-"),
    ("GRN", "-"),
    ("SQE", "-")
]


def make_minicell(scores):
    start = "\\begin{tabular}[x]{@{}c@{}}"
    mid = "\\\\".join(scores)
    end = "\end{tabular}"
    return start + mid + end

def main():
    result_cols = []
    results_path = Path("predicted_scores/cnn_dm_cnn_dm_1000_sample_shuffle_test/")
    for col in ORDER:
        if len(col) == 3:
            column_results = []
            for measure in col[2]:
                abbreviation = measure.split()[1][0].lower()
                system_pred_path = results_path / SYSTEM_FILES[measure]
                scores = read_pairwise_file(system_pred_path)
                acc = compute_pairwise_accuracy(scores)
                column_results.append(f"${acc:.3f}_{{({abbreviation})}}$")
            
            result_cols.append((col[0], make_minicell(column_results), col[1]))
        else:
            system_pred_path = results_path / SYSTEM_FILES[col[0]]
            scores = read_pairwise_file(system_pred_path)
            acc = compute_pairwise_accuracy(scores)
            result_cols.append((col[0], f"{acc:.3f}", col[1]))
    
    table_rows = []
    HEADERS = ["Corpus", "C/D", "WSJ"]
    for idx in range(3):
        col_cells = [c[idx] for c in result_cols]
        table_rows.append("\t&\t".join(HEADERS[idx:idx + 1] + col_cells))

    table_header = "\\begin{tabular}{|l" + "|c" * len(col_cells) + "|" + "}"

    print(table_header)
    print("\\\\\\hline\n".join(table_rows))
    print("\\end{tabular}")

if __name__ == "__main__":
    main()
