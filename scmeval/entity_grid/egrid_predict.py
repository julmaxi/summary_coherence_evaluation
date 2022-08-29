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


class EntityGridPredictor:
    def __init__(self, model_path: str, brown_path: str, ld_path: Path):
        self.model_path = Path(model_path)
        self.brown_path = Path(brown_path)
        self.env = os.environ.copy()

        self.env["LD_LIBRARY_PATH"] = str(ld_path.absolute())

    def score(self, tagged_parse: str):
        try:
            abs_path = self.model_path.resolve()
            process = subprocess.run(
                ["bin64/Predict", "-f", abs_path],
                capture_output=True,
                env=self.env,
                cwd=str(self.brown_path),
                input=tagged_parse.encode("utf8")
            )
        finally:
            pass

        n_sents = len(tagged_parse.split("\n"))
        n_entities = len(tagged_parse.split("\n")[0].split()) - 1
        # Entity Grid computes exact probabiliities, which grow naturally with more entities
        # and sentences. We thus normalize by the number of terms to 
        # reduce length bias.
        return float(process.stdout.decode("utf8")) / (n_sents * n_entities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("infile", type=Path)
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--brown-path", dest="brown_path", default="browncoherence")
    parser.add_argument("--brown-lib", dest="brown_lib_path", default=Path("browncoherence/lib/gsl-2.6/lib/"), type=Path)
    args = parser.parse_args()

    with open(args.infile) as f, open(args.outfile, "w") as f_out:
        reader = csv.reader(f)
        writer = csv.writer(f_out, dialect="unix")
        predictor = EntityGridPredictor(args.model, args.brown_path, args.brown_lib_path)

        index = next(reader)[:-1]
        writer.writerow(index + ["score"])

        for line in tqdm.tqdm(reader):
            score = predictor.score(line[-1])
            writer.writerow(line[:-1] + [score])



if __name__ == "__main__":
    main()


