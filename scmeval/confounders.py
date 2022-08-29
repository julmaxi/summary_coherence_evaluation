from typing import Dict
import pandas as pd
import random

from collections import defaultdict
import argparse

from .evaluation.evaluate import METRICS
from .evaluation.reader import read_gold_scores, read_summaries



def generate_architecture_data(gold_scores: pd.Series, _summaries: pd.Series):
    transformer_architectures = {
        "T5",
        "GPT-2 (zero shot)",
        "BART",
        "Pegasus",
        "Pegasus (dynamic mix)"
    }

    transformer_mask = gold_scores.index.get_level_values("summarizer").isin(transformer_architectures)
    assert transformer_mask.sum() == len(transformer_architectures) * 100
    out = pd.Series([1.0 for _ in gold_scores], index=gold_scores.index) * transformer_mask
    return out


def generate_capitalization_data(_gold_scores: pd.Series, summaries: pd.Series):
    return summaries.map(lambda x: sum(c.isupper() for c in x))


def generate_mean_score_data(gold_scores: pd.Series, _summaries: pd.Series):
    return pd.Series([0 for _ in gold_scores], index=gold_scores.index) + gold_scores.groupby("summarizer").mean()


def get_randomized_metrics(gold_scores: pd.Series, base_data: pd.Series, randomization_range=1e-10, n_trials=100):
    results = defaultdict(list)
    for _ in range(n_trials):
        rand_data = randomize_metric(base_data, randomization_range=randomization_range)
        for metric, result in get_metrics(gold_scores, rand_data).items():
            results[metric] = result    
    return results


def randomize_metric(base_data: pd.Series, randomization_range=1e-10) -> pd.Series:
    return base_data + [random.random() * randomization_range for _ in base_data]


generators = [
    ("cap", generate_capitalization_data),
    ("arc", generate_architecture_data),
    ("max", generate_mean_score_data)
]


def get_metrics(gold_scores: pd.Series, pred_scores: pd.Series) -> Dict[str, float]:
    out = {}
    for metric, func in METRICS:
        result = func(gold_scores, pred_scores)
        out[metric] = result
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--scores", dest="gold_scores", default="data/sumeval_full_scores.csv")
    parser.add_argument("-s", "--summaries", dest="summaries", default="data/sumeval_full_summaries.csv")
    parser.add_argument("--metric", default="Qcoherence")

    args = parser.parse_args()
    gold_scores = read_gold_scores(args.gold_scores, metric=args.metric)
    summaries = read_summaries(args.summaries)
    
    for name, generator in generators:
        data = generator(gold_scores, summaries)
        non_rand_scores = get_metrics(gold_scores, data)
        rand_scores = get_randomized_metrics(gold_scores, data)

        print(name)
        print("Non-Rand")
        for metric, score in non_rand_scores.items():
            print(metric, f"{score:.3f}")

        print("Rand")
        for metric, score in rand_scores.items():
            print(metric, f"{score:.3f}")

        print()





if __name__ == "__main__":
    main()
