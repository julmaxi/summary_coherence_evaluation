import csv
from pathlib import Path

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=Path)
    parser.add_argument("out_dir", type=Path)

    args = parser.parse_args()

    for fpath in args.in_dir.iterdir():
        with open(fpath) as f:
            out_path = args.out_dir / fpath.name
            print(out_path)
            with open(out_path, "w") as f_out:
                reader = csv.reader(f)
                writer = csv.writer(f_out, dialect=reader.dialect)

                header = next(reader)
                print(header)
                replace_idx = header.index("summarizer")
                writer.writerow(header)

                for line in reader:
                    if line[replace_idx] == "M13":
                        line[replace_idx] = "ROUGESal"
                    elif line[replace_idx] == "onmt_pg":
                        line[replace_idx] = "PG"
                    writer.writerow(line)


if __name__ == "__main__":
    main()
