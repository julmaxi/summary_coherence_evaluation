import csv
from typing import Union, Generator, Dict, List, Tuple
from pathlib import Path
from collections import OrderedDict


def read_grids(path: Union[Path, str]) -> Generator[Tuple[OrderedDict[str, str], Dict[str, List[str]]], None, None]:
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            index = OrderedDict()
            for idx, index_name in enumerate(header[:-1]):
                index[index_name] = line[idx]
            grid = parse_entity_grid(line[-1])
            if len(grid) == 0:
                continue
            yield (index, grid)    


def parse_entity_grid(text: str) -> Dict[str, List[str]] :
    out = {}
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        parts = line.split()
        entity = parts[0]
        occurences = parts[1:]

        out[entity] = occurences
    return out
