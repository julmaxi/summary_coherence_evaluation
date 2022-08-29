from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from typing import Dict, List, OrderedDict

from scmeval.evaluation.evaluate import compute_individual_intra_system_correlations, resample_series
from scmeval.evaluation.reader import get_measure_summeval_path, read_gold_scores, read_pred_file




@dataclass
class MeasureSummarizerScoreRange:
    score: float
    lb: float
    ub: float

    @staticmethod
    def from_scores(mean: float, scores: List[float]) -> "MeasureSummarizerScoreRange":
        scores = np.array(scores)
        return MeasureSummarizerScoreRange(
            mean,
            np.quantile(scores, 0.025),
            np.quantile(scores, 0.975)
        )


def plot_intra_system_correlation_for_measures(names: List[str]):
    names = reversed(names)
    gold_scores = read_gold_scores("data/sumeval_full_scores.csv", "Qcoherence")
    measure_score_ranges = OrderedDict()
    for name in names:
        pred_scores = read_pred_file(get_measure_summeval_path(name), None, list(gold_scores.index.names))
        measure_score_ranges[name] = compute_summarizer_score_ranges(gold_scores, pred_scores)
    
    plot_intra_summarizer_correlation_bounds(gold_scores, measure_score_ranges)


def compute_summarizer_score_ranges(gold_scores: pd.Series, pred_scores: pd.Series, n_resamples=1000) -> Dict[str, MeasureSummarizerScoreRange]:
    summarizer_correlations = defaultdict(list)
    best_est_corr = compute_individual_intra_system_correlations(gold_scores, pred_scores)

    means = {}
    for summarizer, correlation in best_est_corr.items():
        means[summarizer] = correlation

    for _ in range(n_resamples):
        resampled_gold, resampled_pred = resample_series(gold_scores, pred_scores, exclude=["summarizer"])
        for summarizer, correlation in compute_individual_intra_system_correlations(resampled_gold, resampled_pred).items():
            summarizer_correlations[summarizer].append(correlation)
    
    return {
        k: MeasureSummarizerScoreRange.from_scores(means[k], v)
        for k, v in summarizer_correlations.items()
    }


def plot_intra_summarizer_correlation_bounds(gold_scores: pd.Series, score_ranges: Dict[str, Dict[str, MeasureSummarizerScoreRange]]):
    plt.rcParams['figure.figsize'] = [14, 8]
    dist = 3.5

    #print(score_ranges)
    all_summarizers = (-gold_scores.groupby("summarizer").mean()).sort_values().index.unique("summarizer")
    summarizer_order = list(all_summarizers)
    gs = gridspec.GridSpec(len(all_summarizers) // 2 + 1, 2, width_ratios=[1, 1],
            wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845) 

    flat_axis = []
    
    for y in range(2):
        for x in range(len(all_summarizers) // 2 + 1):
            if x == 17//2 and y==1:
                continue
            ax = plt.subplot(gs[x,y])
            flat_axis.append(ax)
            
            ax.axes.get_yaxis().set_visible(False)
            
            if not (
                (x == len(all_summarizers) // 2 and y == 0)
                or (x == len(all_summarizers) // 2 - 1 and y == 1)
                ):
                ax.axes.get_xaxis().set_visible(False)
                
            val = summarizer_order[(17//2 + 1) * y + x]
            ax.text(-0.42, 2.0 * len(score_ranges) - 2.5, val)

    for idx, (measure, measure_score_ranges) in enumerate(score_ranges.items()):
        for sys_idx, system in enumerate(summarizer_order):
            score_range = measure_score_ranges[system]
            sys_lb = score_range.lb
            sys_ub = score_range.ub
            sys_mean = score_range.score
            
            ax = flat_axis[sys_idx] 
            
            ax.set_xlim(-0.45, 1.0)
            ax.set_ylim(-2.0, 2.0 * len(score_ranges) + 1.0)
            
            ax.errorbar(sys_mean, idx * 2, xerr=np.array([[sys_mean - sys_lb, sys_ub - sys_mean]]).T, fmt='o', capsize=3, label=measure)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc="lower right")


