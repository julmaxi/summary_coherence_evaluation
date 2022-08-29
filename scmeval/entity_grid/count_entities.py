import argparse
from collections import defaultdict
import csv
from pathlib import Path
from typing import Counter, Dict, List, Tuple, Union, Generator
import itertools as it

from pandas import cut

from .reader import read_grids

def count_entity_frequences(grid: Dict[str, List[str]]) -> Counter:
    occurence_counts = Counter()
    for occurences in grid.values():
        num_occurences = sum((1 if o != "-" else 0) for o in occurences)
        occurence_counts[num_occurences] += 1

    return occurence_counts


def count_max_entity_freq(grid_generator):
    all_freqs = [count_entity_frequences(grid[1]) for grid in grid_generator]
    max_freqs = [max(c.keys()) for c in all_freqs]
    max_freq_counts = Counter(max_freqs)

    return max_freq_counts


def count_num_frequent_entities(grid_generator):
    all_freqs = [count_entity_frequences(grid[1]) for grid in grid_generator]

    num_ents = [sum(f.values()) - f[1] for f in all_freqs]

    return Counter(num_ents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("grids", nargs="+")
    parser.add_argument("-c", "--cutoff", default=None, type=int)
    
    args = parser.parse_args()

    grid_generator = it.chain(*(read_grids(fname) for fname in args.grids))
    max_freq_counts = count_num_frequent_entities(grid_generator)

    total = sum(max_freq_counts.values())

    max_freq_counts = sorted(max_freq_counts.items())
    max_rel_freqs = [(k, v / total) for k, v in max_freq_counts]

    if args.cutoff:
        limited_rel_freqs = []
        cutoff_total = 0.0
        for key, val in max_rel_freqs:
            if key < args.cutoff:
                limited_rel_freqs.append((key, val))
            else:
                cutoff_total += val
        
        limited_rel_freqs.append((f"{args.cutoff}+", cutoff_total))
        max_rel_freqs = limited_rel_freqs
                


    for key, val in max_rel_freqs:
        print(key, f"{val:.3f}")


if __name__ == "__main__":
    main()
