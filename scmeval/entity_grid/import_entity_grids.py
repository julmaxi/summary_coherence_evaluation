import csv
import argparse

import subprocess
import tempfile

import os
import re
from xml.parsers.expat import model
import glob
import tqdm
from pathlib import Path


"""
This imports entity grids in a directory into a csv
file for easier processing.
Use this for externally parsed grids.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=Path)
    parser.add_argument("--grids-dir", dest="grids_dir", default=Path("outputs/entity-grids"), type=Path)
    parser.add_argument("-n", "--name", dest="out_name", default=None)

    args = parser.parse_args()

    out_name = args.out_name

    if out_name is None:
        out_name = args.indir.name

    with open(args.grids_dir / out_name, "w") as f:
        writer = csv.writer(f, dialect="unix")
        writer.writerow(["id", "grid"])
        for path in args.indir.iterdir():
            file_id = path.name.rsplit(".", 1)[0]
            writer.writerow([file_id, path.read_text()])


if __name__ == "__main__":
    main()
