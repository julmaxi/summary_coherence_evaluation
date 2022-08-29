import argparse
import itertools as it
from scmeval.entity_grid.entity_graph import create_graph_from_grid

from scmeval.entity_grid.reader import read_grids

import networkx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("grids", nargs="+")
    parser.add_argument("-c", "--cutoff", default=None, type=int)
    
    args = parser.parse_args()

    grid_generator = it.chain(*(read_grids(fname) for fname in args.grids))

    
    n_unconnected = []
    for _, grid in grid_generator:
        graph = create_graph_from_grid(grid)
        n_unconnected.append(
            len([n for (n, d) in graph.degree(weight="weight") if d == 0]) / len(graph)
        )

    unconnected_freq = sum(n_unconnected) / len(n_unconnected)
    print(f"{unconnected_freq:.3f}")


if __name__ == "__main__":
    main()
