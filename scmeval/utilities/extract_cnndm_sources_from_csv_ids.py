import sys

import pandas as pd
import datasets

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("docs")
    parser.add_argument("outfile")
    args = parser.parse_args()

    df = pd.read_csv(args.docs)

    if "doc" in df.columns:
        doc_column = "doc"
    elif "id" in df.columns:
        doc_column = "id"

    docs = set(df[doc_column])

    ds = datasets.load_dataset("cnn_dailymail", "3.0.0")

    entries = ds.filter(
        lambda x: x["id"] in docs
    ).map(lambda x: {"doc": x["id"]})

    entries = {x["id"]: x["article"] for s in entries.values() for x in s}

    out = []
    df.set_index(list(df.columns[:-1]), inplace=True)

    for doc_id in docs:
        out.append({f"{doc_column}": doc_id, "source": entries[doc_id]})

    out = pd.DataFrame.from_records(out, columns=[doc_column, "source"], index=doc_column)
    out.to_csv(args.outfile)
