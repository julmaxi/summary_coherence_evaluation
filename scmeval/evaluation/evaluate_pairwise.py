import argparse
from matplotlib.cbook import index_of
import pandas as pd
import sys
from typing import List, Tuple
import itertools as it
import random



def read_pairwise_file(path, predictor=None):
    scores = pd.read_csv(path, index_col=["id", "label"])
    if predictor is None:
        if len(scores.columns) > 1:
            print("Warning, no column specificed and more than one column in dataset. Using first column.", file=sys.stderr)
        
        scores = scores[scores.columns[0]]
    else:
        scores = scores[predictor]
    
    return scores


def compute_pairwise_accuracy(scores):
    total = 0
    wins = 0
    for idx, pairs in scores.groupby("id"):
        pos = pairs.loc[idx, 1].item()
        neg = pairs.loc[idx, 0].item()

        wins += int(pos > neg) + int(pos == neg) * 0.5
        total += 1
    return wins / total


def shuffled(t):
    l = list(t)
    random.shuffle(l)
    return tuple(l)


def compute_inter_doc_shuffle_accurcy_length_curve(scores: pd.Series, lengths: pd.Series, n_trials=1000) -> pd.DataFrame:
    sample_points = []

    potential_combinations = list(it.combinations(scores.index.unique("id"), 2))
    random.shuffle(potential_combinations)
    potential_combinations = [shuffled(p) for p in potential_combinations[:n_trials]]

    for (l_doc, r_doc) in potential_combinations:
        pos_score = scores.loc[l_doc, 1]
        neg_score = scores.loc[r_doc, 0]

        length_diff = lengths[l_doc, 1] - lengths[r_doc, 0]

        sample_points.append((length_diff, pos_score > neg_score))
    
    return pd.DataFrame.from_records(sample_points, columns=["length_diff", "correct"])




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument( "-p", "--predictor", default=None)

    args = parser.parse_args()

    scores = read_pairwise_file(args.file, args.predictor)
    score = compute_pairwise_accuracy(scores)
    
    print(f"{score:.3f}")


if __name__ == "__main__":
    main()
