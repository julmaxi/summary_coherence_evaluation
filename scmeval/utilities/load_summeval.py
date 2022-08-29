import json
import sys
from pathlib import Path

import pandas as pd
import random

import argparse


def read_sumqe_experts(f, source_dir, ref_dir):
    result = []
    for line in f:
        entry = json.loads(line)

        for idx, annotation in enumerate(entry["expert_annotations"]):
            row = {}
            for key, val in annotation.items():
                row["Q" + key] = val

            row["assessor"] = "A" + str(idx)
            row["doc"] = entry["id"]
            row["summarizer"] = entry["model_id"]
            row["summary"] = entry["decoded"]

            basename = entry["id"].rsplit("-", 1)[-1]

            row["source"] = (source_dir / f"{basename}").read_text()
            row["reference"] = (ref_dir / f"{basename}").read_text()

            result.append(row)
    
    return pd.DataFrame.from_records(result, index=["doc", "summarizer", "assessor"])


def write_df(df, prefix):
    summaries = df[["summary"]].xs("A1", level="assessor")
    
    summaries.to_csv(prefix + "_summaries.csv")

    df.drop('summary', axis=1).drop('source', axis=1).drop('reference', axis=1).to_csv(prefix + "_scores.csv")
    df[["source"]].xs("A1", level="assessor").xs("BART", level="summarizer").to_csv(prefix + "_source.csv")
    df[["reference"]].xs("A1", level="assessor").xs("BART", level="summarizer").to_csv(prefix + "_reference.csv")

def replace_system_names(index):
    names = {
            "M0": "LEAD-3",
            "M1": "neusum",
            "M2": "BanditSum",
            "M5": "RNES",
            "M8": "onmt_pg",
            "M9": "abssentrw",
            "M10": "Bottom-Up",
            "M11": "Improve-abs",
            "M12": "Unified-ext-abs",
            "M13": "ROUGESal",
            "M14": "Multi-task (Ent + QG)",
            "M15": "Closed book decoder",
            "M17": "T5",
            "M20": "GPT-2 (zero shot)",
            "M22": "BART",
            "M23": "Pegasus",
            "M23_dynamicmix": "Pegasus (dynamic mix)"
    }
    return names.get(index, index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summeval_path", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("reference_dir", type=Path)

    args = parser.parse_args()

    with open(args.summeval_path) as f:
        df = read_sumqe_experts(f, args.source_dir, args.reference_dir)
        summarizers = set(df.index.to_frame()["summarizer"].unique())
        test_inst = set(random.sample(summarizers, 5))
        train_inst = summarizers - test_inst

        test_dfs = []
        for key in test_inst:
            test_dfs.append(df.xs(key, level="summarizer", drop_level=False))
        test_df = pd.concat(test_dfs)

        test_df.rename(replace_system_names, inplace=True)

        train_dfs = []
        for key in train_inst:
            train_dfs.append(df.xs(key, level="summarizer", drop_level=False))
        train_df = pd.concat(train_dfs)
        train_df.rename(replace_system_names, inplace=True)
        Path("data").mkdir(exist_ok=True)

        write_df(pd.concat(train_dfs + test_dfs).rename(replace_system_names), "data/sumeval_full")
