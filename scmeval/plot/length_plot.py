from datetime import datetime
from random import random
from typing import Iterable, List, Any, Set, Union, Tuple
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import math

import numpy as np

from scmeval.evaluation.evaluate_pairwise import compute_inter_doc_shuffle_accurcy_length_curve


from ..evaluation.reader import get_measure_summeval_path, read_gold_scores, read_pred_file, read_summaries, read_measure_cnn_dm_shuffle_results
from ..evaluation.measure_bias_matrix import compute_measure_bias_matrix


def normalize(s):
    return (s - s.mean()) / s.std()


def plot_measure_length_correlation(names: List[str]):
    gold_scores = read_gold_scores("data/sumeval_full_scores.csv", "Qcoherence")
    summaries = read_summaries("data/sumeval_full_summaries.csv")
    parses = pd.read_csv("outputs/stanford-parses/sumeval_full_summaries.csv", index_col=list(gold_scores.index.names))
    lengths = parses[parses.columns[0]].map(lambda x: len(x.split("\n"))).rename("length")
    #import spacy
    #nlp = spacy.load("en_core_web_sm")
    #lengths = summaries.map(lambda x: len(list(nlp(x).sents))).rename("length")
    #lengths = parses[parses.columns[0]].map(lambda x: 10.0 * math.floor(min(100, max(20, len(x.split(" ")))) / 10)).rename("length")

    gold_scores = normalize(gold_scores)
    df = lengths.to_frame().join(gold_scores.rename("score")).groupby("length").mean().reset_index()
    plt.plot(df["length"], df["score"], label="gold")

    for name in names:
        pred_path = get_measure_summeval_path(name)
        pred_scores = read_pred_file(pred_path, None, index_cols=list(gold_scores.index.names))
        pred_scores = normalize(pred_scores)
        df = lengths.to_frame().join(pred_scores.rename("score")).groupby("length").mean().reset_index()
        plt.plot(df["length"], df["score"], label=name)
    
    plt.legend()


def plot_measure_length_sensitivty(names):
    orig_texts = pd.read_csv("data/cnn_dm_1000_sample_shuffle_test.csv", index_col=["id", "label"])["text"]
    lengths = orig_texts.map(len)

    plt.gca().set_prop_cycle(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], marker=['o', '+', 'x', '*'])
    for name in names:
        scores = read_measure_cnn_dm_shuffle_results(name)
        
        samples = compute_inter_doc_shuffle_accurcy_length_curve(scores, lengths, n_trials=10000)

        #samples = np.array(samples)
        samples["length_diff"] = samples["length_diff"].map(lambda x: max(min(math.floor(x / 20) * 20, 200), -200))
        samples = samples.groupby("length_diff").mean().reset_index()
        plt.plot(samples["length_diff"], samples["correct"], label=name)
    plt.legend()
    plt.xlabel("Length difference between shuffled and unshuffled documents\nShuffled longer $\longleftarrow$ $\longrightarrow$ Unshuffled longer")
    plt.ylabel("Accuracy")


def get_ranking_length_samples(correct_pairs: List[Tuple[str, str]], lengths: pd.Series, scores: pd.Series) -> pd.DataFrame:
    samples = []
    import random
    random.shuffle(correct_pairs)
    for l, r in correct_pairs[:10000]:
        samples.append((lengths.loc[l] - lengths.loc[r], scores.loc[l] > scores.loc[r]))
    
    return pd.DataFrame.from_records(samples, columns=["length_diff", "correct"])


def get_correct_pairs(scores: pd.Series) -> List[Tuple[str, str]]:
    pairs = []

    import itertools as it
    for ((idx_1, score_1), (idx_2, score_2)) in it.combinations(scores.iteritems(), 2):
        if score_1 > score_2:
            pairs.append((idx_1, idx_2))
        elif score_1 < score_2:
            pairs.append((idx_2, idx_1))
    
    return pairs



def plot_measure_summeval_length_sensitivty(names):
    summaries = pd.read_csv("data/sumeval_full_summaries.csv", index_col=["doc", "summarizer"])["summary"]
    gold_scores = read_gold_scores("data/sumeval_full_scores.csv", "Qcoherence")
    pairs = get_correct_pairs(gold_scores)
    lengths = summaries.map(len)

    plt.gca().set_prop_cycle(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], marker=['o', '+', 'x', '*'])
    for name in names:
        scores = read_pred_file(get_measure_summeval_path(name), None, list(gold_scores.index.names))
        
        samples = get_ranking_length_samples(pairs, lengths, scores)

        #samples = np.array(samples)
        samples["length_diff"] = samples["length_diff"].map(lambda x: max(min(math.floor(x / 20) * 20, 200), -200))
        #samples["length_diff"] = samples["length_diff"].map(lambda x: math.floor(x / 20) * 20)
        samples = samples.groupby("length_diff").mean().reset_index()
        plt.plot(samples["length_diff"], samples["correct"], label=name)

    plt.legend()
    plt.xlabel("Length Diff")
    plt.ylabel("Accuracy")


