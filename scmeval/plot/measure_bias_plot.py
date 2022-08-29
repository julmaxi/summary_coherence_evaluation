from typing import Iterable, List, Any, Union, Tuple
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import math

import numpy as np

from ..evaluation.reader import get_measure_summeval_path, read_gold_scores, read_pred_file
from ..evaluation.measure_bias_matrix import compute_measure_bias_matrix


def flatten(l: Iterable[Any]):
    for x in l:
        if hasattr(x, "__iter__"):
            for y in flatten(x):
                yield y
        else:
            yield x


def plot_measures_matrices(names: List[Union[str, Tuple[str, pd.Series]]]):
    n_rows = 2
    n_cols = int(math.ceil(len(names) / n_rows))
    fig, axis = plt.subplots(n_rows, n_cols)

    fig.set_size_inches(10.0, 6.0)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    gold_scores = read_gold_scores("data/sumeval_full_scores.csv", "Qcoherence")
    measure_matrices = OrderedDict()
    for name in names:
        if isinstance(name, str):
            pred_path = get_measure_summeval_path(name)
            pred_scores = read_pred_file(pred_path, None, index_cols=list(gold_scores.index.names))
        else:
            name, pred_scores = name
        measure_matrices[name] = compute_measure_bias_matrix(gold_scores, pred_scores)

    for idx, (ax, (name, (matrix, population))) in enumerate(zip(flatten(axis), measure_matrices.items())):
        plt.sca(ax)
        plt.title(name)
        patches = plot_measure_bias_matrix(matrix, population, include_xlabels=idx in (3, 4, 5), include_ylabels=idx in (0, 3), include_bar=False)

    plt.colorbar(patches, ax=axis, shrink=0.5)


def plot_measure_bias_matrix(
    matrix: pd.DataFrame,
    populations: pd.DataFrame,
    min_radius=0.00,
    max_radius=0.99,
    include_xlabels=True,
    include_ylabels=True,
    include_bar=True
):
    matrix = matrix.loc[::-1]
    populations = populations.loc[::-1]

    x_labels = list(matrix.columns)
    y_labels = list(matrix.index)
    
    matrix = matrix.to_numpy()
    
    populations = populations.to_numpy()
    ax = plt.gca()
    
    x_coords, y_coords = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    
    norm_pops = populations / populations.max()
    # Scale the area, not the radius
    norm_pops = (norm_pops ** 0.5) * 0.5 * (max_radius - min_radius) + min_radius
    
    markers = [plt.Circle((x, y), radius=p) for x, y, p in zip(x_coords.flat, y_coords.flat, norm_pops.flat)]
    collection = PatchCollection(markers, array=matrix.flatten(), cmap="RdYlGn")
    collection.set_clim(-1.0, 1.0)

    ax.add_collection(collection)
    
    if include_xlabels:
        ax.set(xticks=np.arange(matrix.shape[0]), 
           xticklabels=x_labels)
        ax.tick_params(axis='x', rotation=90)
    else:
        ax.tick_params(
        axis='x',          
        which='both',      
        bottom=False,     
        top=False,         
        labelbottom=False) 


        
    if include_ylabels:
        ax.set(yticks=np.arange(matrix.shape[1]),
            yticklabels=y_labels)
    else:
        ax.tick_params(
        axis='y',          
        which='both',      
        left=False,     
        right=False,         
        labelleft=False) 

    ax.set_xticks(
        np.arange(matrix.shape[0] + 1) - 0.5,
        minor=True,
    )
    ax.set_yticks(
        np.arange(matrix.shape[1] + 1) - 0.5,
        minor=True
    )

    ax.grid(which='minor')
    ax.set_aspect("equal")
    
    if include_bar:
        plt.gcf().colorbar(collection)
    
    return collection


