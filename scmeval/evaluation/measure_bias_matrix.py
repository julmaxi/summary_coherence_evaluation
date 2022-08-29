import itertools as it
from typing import Tuple

import pandas as pd


def compute_measure_bias_matrix(gold_scores: pd.Series, measure_scores: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rankings = []
    populations = []

    mean_gold_scores = gold_scores.groupby("summarizer").mean().to_dict()
    measure_order = sorted(gold_scores.index.unique("summarizer"), key=lambda x: mean_gold_scores[x])

    for sys_1, sys_2 in it.combinations(measure_order, 2):
        gold_1 = gold_scores.xs(sys_1, level="summarizer").sort_index()
        gold_2 = gold_scores.xs(sys_2, level="summarizer").sort_index()
        pred_1 = measure_scores.xs(sys_1, level="summarizer").sort_index()
        pred_2 = measure_scores.xs(sys_2, level="summarizer").sort_index()
   
        n_correct = 0
        n_correct_better = 0
        n_correct_worse = 0
        n_total = 0
        n_total_better = 0
        n_total_worse = 0
    
        for (g1, g2, p1, p2) in zip(gold_1, gold_2, pred_1, pred_2):
            if g1 == g2:
                continue
        
            if g1 > g2:
                n_correct_better += (p1 > p2)
                n_total_better += 1
            else:
                n_correct_worse += (p1 < p2)
                n_total_worse += 1
        
            n_correct += (g1 < g2) == (p1 < p2)
            n_total += 1
    
        tau_plus = n_correct_better / n_total_better - ((n_total_better - n_correct_better) / n_total_better)
        tau_minus = n_correct_worse / n_total_worse - ((n_total_worse - n_correct_worse) / n_total_worse)
        pop_plus = n_total_better
        pop_minus = n_total_worse
        
        rankings.append({
            "sys_1": sys_1,
            "sys_2": sys_2,
            "tau": tau_plus
        })
        rankings.append({
            "sys_2": sys_1,
            "sys_1": sys_2,
            "tau": tau_minus
        })
        
        populations.append({
            "sys_1": sys_1,
            "sys_2": sys_2,
            "pop": pop_plus
        })
        populations.append({
            "sys_2": sys_1,
            "sys_1": sys_2,
            "pop": pop_minus
        })

    bias_matrix = pd.DataFrame.from_records(rankings).set_index(["sys_1", "sys_2"]).unstack().fillna(0.0)
    bias_matrix.columns = bias_matrix.columns.droplevel()
    bias_matrix = bias_matrix.sort_index(key=lambda i: [-mean_gold_scores[x] for x in i])
    bias_matrix = bias_matrix.sort_index(key=lambda i: [-mean_gold_scores[x] for x in i], axis=1)
    
    populations = pd.DataFrame.from_records(populations).set_index(["sys_1", "sys_2"]).unstack().fillna(0.0)
    populations.columns = populations.columns.droplevel()
    
    populations = populations.loc[bias_matrix.index]
    populations = populations[bias_matrix.columns]

    return bias_matrix, populations
