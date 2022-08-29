import argparse
import pandas as pd
from genericpath import exists
from pathlib import Path
import re
#from nltk import sent_tokenize
import random
import itertools as it
import numpy as np
import json


def parse_bracket_file(text):
    def parse(token_iter):
        out = []

        while True:
            next_tok = next(token_iter, None)
            #print(next_tok)
            if next_tok is None:
                break
            elif next_tok == ")":
                break
            elif next_tok == "(":
                out.append(parse(token_iter))
            else:
                out.append(next_tok)
        
        return out

    tokens = re.findall("(\(|\)|[^)() \t\n]+)", text)

    tok_iter = iter(tokens)
    sents = []
    while True:
        next_tok = next(tok_iter, None)
        if next_tok is None:
            break
        assert next_tok == "("
        out = parse(tok_iter)
        if len(out) == 0:
            break
        sents.append(out)
    
    return sents


def parse_to_text(tree):
    if isinstance(tree[1], list):
        out = []
        for elem in tree[1:]:
            out.append(parse_to_text(elem))
        
        return " ".join(out)
    else:
        return tree[1]


def parse_bracket_file(text):
    def parse(token_iter):
        out = []

        while True:
            next_tok = next(token_iter, None)
            #print(next_tok)
            if next_tok is None:
                break
            elif next_tok == ")":
                break
            elif next_tok == "(":
                out.append(parse(token_iter))
            else:
                out.append(next_tok)
        
        return out

    tokens = re.findall("(\(|\)|[^)() \t\n]+)", text)

    tok_iter = iter(tokens)
    sents = []
    while True:
        next_tok = next(tok_iter, None)
        if next_tok is None:
            break
        assert next_tok == "("
        out = parse(tok_iter)
        if len(out) == 0:
            break
        sents.append(out)
    
    return sents


def parse_to_text(tree):
    if isinstance(tree[1], list):
        out = []
        for elem in tree[1:]:
            out.append(parse_to_text(elem))
        
        return " ".join(out)
    else:
        return tree[1]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("summaries", type=Path)
    parser.add_argument("parses", type=Path)
    parser.add_argument("grids", type=Path)
    parser.add_argument("out_path", type=Path)

    args = parser.parse_args()


    summaries = pd.read_csv(args.summaries)
    summaries.set_index(list(summaries.columns[:-1]), inplace=True)

    parses = pd.read_csv(args.parses)
    parses.set_index(list(parses.columns[:-1]), inplace=True)
    parses = parses[parses.columns[0]]

    grids = pd.read_csv(args.grids)
    grids.set_index(list(grids.columns[:-1]), inplace=True)
    grids = grids[grids.columns[0]]

    assert len(summaries) == len(parses) == len(grids)

    out = []
    for idx, (index, row) in enumerate(summaries.iterrows()):
        grid = grids.loc[index]
        grid = [l.strip().split(" ") for l in grid.strip().split("\n")]
        parse = parses.loc[index]
        
        sentences = list(map(parse_to_text, parse_bracket_file(parse)))
        if len(sentences) != len(grid[0]) - 1:
            raise RuntimeError(f"{len(sentences)} != {len(grid[0]) - 1}")

        grid_text = "\n".join(map(lambda x: " ".join(x), grid))
        out.append({
            "sents": sentences,
            "grid": grid_text,
            "index": list(zip(summaries.index.names, index))
        })
    with open(args.out_path, "w") as f:
        print(len(out))
        json.dump(out, f)

if __name__ == "__main__":
    main()
