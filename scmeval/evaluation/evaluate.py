from dataclasses import dataclass
from functools import partial
from operator import index
import random
from typing import Callable, Dict, List, Union
import scipy.stats
import pandas as pd
import itertools as it
from typing import Set
from collections import namedtuple
import argparse
import sys
import numpy as np
from pathlib import Path

from .reader import read_gold_scores, read_pred_file


def tau_a_mod(gold: np.ndarray, pred: np.ndarray):
    a_cmp = np.sign(gold[:,None] - gold[None,:])
    b_cmp = np.sign(pred[:,None] - pred[None,:])
    
    comparison_idx = (a_cmp != 0).nonzero()
    a_cmp = a_cmp[comparison_idx]
    b_cmp = b_cmp[comparison_idx]

    conc = (a_cmp == b_cmp).sum() / 2
    disc = (a_cmp != b_cmp).sum() / 2

    n = len(a_cmp)

    return (conc - disc) / (n / 2)


def compute_system_level_correlation(gold_scores: pd.Series, pred_scores: pd.Series) -> float:
    return scipy.stats.kendalltau(
        gold_scores.groupby("summarizer", sort=True).mean(),
        pred_scores.groupby("summarizer", sort=True).mean()
    ).correlation


def compute_document_level_correlation(gold_scores: pd.Series, pred_scores: pd.Series) -> float:
    return scipy.stats.kendalltau(
        gold_scores.sort_index(),
        pred_scores.sort_index()
    ).correlation


def compute_intra_system_correlation(gold_scores: pd.Series, pred_scores: pd.Series) -> float:
    partial_scores = compute_individual_intra_system_correlations(gold_scores, pred_scores)
    return sum(partial_scores.values()) / len(partial_scores)


def compute_individual_intra_system_correlations(gold_scores: pd.Series, pred_scores: pd.Series) -> Dict[str, float]:
    joint_scores = gold_scores.rename("gold").to_frame().join(pred_scores.rename("pred"))

    partial_scores = {}
    for key, scores in joint_scores.groupby("summarizer"):
        partial_scores[key] = scipy.stats.kendalltau(
            scores["gold"].to_numpy(),
            scores["pred"].to_numpy()
        ).correlation
    return partial_scores


def compute_intra_document_correlation(gold_scores: pd.Series, pred_scores: pd.Series) -> float:
    joint_scores = gold_scores.rename("gold").to_frame().join(pred_scores.rename("pred"))

    partial_scores = {}
    for key, scores in joint_scores.groupby("doc"):
        partial_scores[key] = scipy.stats.kendalltau(scores["gold"], scores["pred"]).correlation
    
    return sum(partial_scores.values()) / len(partial_scores)


RankingEntry = namedtuple("RankingEntry", "doc better worse")


def compute_rankings(scores: pd.Series) -> Set[RankingEntry]:
    #docwise_scores = .apply(list)

    rankings = set()

    for doc, doc_scores in scores.groupby("doc", sort=False):
        for (l_summarizer, l_score), (r_summarizer, r_score) in it.combinations(doc_scores.iteritems(), 2):
            if l_score > r_score:
                rankings.add(RankingEntry(doc, l_summarizer, r_summarizer))
            elif r_score > l_score:
                rankings.add(RankingEntry(doc, r_summarizer, l_summarizer))
    
    return rankings


def compute_pairwise_ranking_accuracy(gold_scores: pd.Series, pred_scores: pd.Series) -> float:
    gold_rankings = compute_rankings(gold_scores)
    return len(gold_rankings & compute_rankings(pred_scores)) / len(gold_rankings)


@dataclass
class MetricResult:
    score: float
    resampling_result: List[float]
    ci_95_lower_bound: float
    ci_95_upper_bound: float
    

def compute_metric(measure : Callable[[pd.Series, pd.Series], float], gold_scores: pd.Series, pred_scores: pd.Series, n_resample=100) -> MetricResult:
    score = measure(gold_scores, pred_scores)

    if n_resample > 0:
        resample_results = []
        for _ in range(n_resample):
            sampled_gold, sampled_pred = resample_series(gold_scores, pred_scores)
            sampled_score = measure(sampled_gold, sampled_pred)

            if not np.isnan(sampled_score):
                resample_results.append(sampled_score)
        
        lb = np.quantile(resample_results, 0.025)
        ub = np.quantile(resample_results, 0.975)
        
    else:
        resample_results = None
        lb = None
        ub = None

    return MetricResult(score, resample_results, lb, ub)


METRICS = [
    ("sys", compute_system_level_correlation),
    ("doc", compute_document_level_correlation),
    ("intra", compute_intra_system_correlation),
    ("intrad", compute_intra_document_correlation),
    ("pair", compute_pairwise_ranking_accuracy)
]


def resample_series(*series: pd.Series, exclude=None) -> pd.Series:
    series0 = series[0]

    sampled_indices = []
    is_excluded = []
    for index_name in series0.index.names:
        if exclude is not None and index_name in exclude:
            sampled_indices.append(series0.index.unique(index_name))
            is_excluded.append(True)
        else:
            values = series0.index.unique(index_name)
            samples = random.choices(values, k=len(values))
            sampled_indices.append([s for s in samples])
            is_excluded.append(False)
            
    out_series = []
    for old_series in series:
        new_data = np.empty(len(series0))
        new_index = np.empty((len(series0.index.names), len(series0)), dtype=object)
    
        for idx, sampled_index in enumerate(it.product(*map(enumerate, sampled_indices))):
            source_index = tuple(s[1] for s in sampled_index)
            target_index = [(s[0] if not is_ex else s[1]) for s, is_ex in zip(sampled_index, is_excluded)]
            for index_idx, elem in enumerate(target_index):
                new_index[index_idx, idx] = str(elem)

            new_data[idx] = old_series.loc[source_index]
        
        new_series = pd.Series(new_data, index=pd.MultiIndex.from_arrays(new_index, names=series0.index.names))
        out_series.append(new_series)
    
    return out_series


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("gold_scores")
    parser.add_argument("pred_scores")
    parser.add_argument("--metric", default="Qcoherence")
    parser.add_argument("-p", "--predictor", default=None)
    parser.add_argument("-r", "--resample", dest="n_resample", default=0, type=int)

    args = parser.parse_args()

    gold_scores = read_gold_scores(args.gold_scores, metric=args.metric)
    pred_scores = read_pred_file(args.pred_scores, args.predictor, list(gold_scores.index.names))
            
    for metric, func in METRICS:
        result = compute_metric(func, gold_scores, pred_scores, n_resample=args.n_resample)

        if args.n_resample > 0:
            print(f"{metric}: {result.score:.3f} ({result.ci_95_lower_bound:.3f}, {result.ci_95_upper_bound:.3f})")
        else:
            print(f"{metric}: {result.score:.3f}")


if __name__ == "__main__":
    main()
