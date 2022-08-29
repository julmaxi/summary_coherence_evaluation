import argparse
from collections import defaultdict
from typing import Dict, List
import itertools as it

from networkx import DiGraph
import csv
from .reader import read_grids



WEIGHTS = {
    "S": 3,
    "O": 2,
    "X": 1
}

def create_graph_from_grid(grid: Dict[str, List[str]]) -> DiGraph:
    n_sentences = len(next(iter(grid.values())))
    sentence_entities : Dict[int, Dict[int, str]] = defaultdict(dict)
    for entity_idx, (_, occurences) in enumerate(sorted(grid.items())):
        for sent_idx, occurence in enumerate(occurences):
            if occurence != "-":
                sentence_entities[sent_idx][entity_idx] = occurence

    graph = DiGraph()
    graph.add_nodes_from(range(n_sentences))
    for l_sent, r_sent in it.combinations(range(n_sentences), 2):
        l_entities = sentence_entities[l_sent]
        r_entities = sentence_entities[r_sent]

        link_weight = 0
        for l_entity, l_role in l_entities.items():
            r_role = r_entities.get(l_entity)
            if r_role is None:
                continue

            link_weight += WEIGHTS[l_role] * WEIGHTS[r_role]
        
        graph.add_edge(l_sent, r_sent, weight=link_weight / (r_sent - l_sent))
    
    return graph


def compute_avg_outdegree(g: DiGraph) -> float:
    return sum(w for _, w in g.out_degree(weight="weight")) / len(g.nodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grids")
    parser.add_argument("outfile")

    args = parser.parse_args()

    grids = read_grids(args.grids)

    with open(args.outfile, "w") as f:
        writer = csv.writer(f, dialect="unix")
        for row_idx, (index, grid) in enumerate(grids):
            graph = create_graph_from_grid(grid)
            score = compute_avg_outdegree(graph)

            if row_idx == 0:
                writer.writerow([k for k in index.keys()] + ["score"])
            
            writer.writerow([v for v in index.values()] + [score])
