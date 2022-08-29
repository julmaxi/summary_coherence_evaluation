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


class StanfordParser:
    def __init__(self, parser_path: str, model_path: str) -> None:
        self.parser_path = parser_path
        self.model_path = model_path

    def parse(self, text) -> str:
        # java -mx16G -cp "$scriptdir/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \ -outputFormat "penn" -writeOutputFiles -outputFilesDirectory $target edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz

        
        fd, name = tempfile.mkstemp()

        try:
            with os.fdopen(fd, "w") as f_temp:
                f_temp.write(text)

            process = subprocess.run([
                "java", "-mx16G", "-cp", f"{self.parser_path}/*", "edu.stanford.nlp.parser.lexparser.LexicalizedParser",
                "-outputFormat", 'penn', self.model_path, name
            ], capture_output=True)
        finally:
            os.remove(name)

        return process.stdout.decode("utf8")


class OpenNLPTagger:
    def __init__(self, jar_path: str, model_path: str) -> None:
        self.jar_path = jar_path
        self.model_path = model_path

    def tag(self, text: str) -> str:
        process = subprocess.run([
            "java", "-mx16G", "-cp", f"{self.jar_path}/*", "opennlp.tools.lang.english.NameFinder", "-parse",
            *glob.glob(f"{self.model_path}/*.bin")
        ], capture_output=True, input=text.encode("utf8"))

        return process.stdout.decode("utf8")


class EntityGridBuilder:
    def __init__(self, brown_path: str, ld_path: Path):
        self.brown_path = Path(brown_path)
        self.env = os.environ.copy()

        self.env["LD_LIBRARY_PATH"] = str(ld_path.absolute())

    def create_grid(self, tagged_parse: str):
        fd, name = tempfile.mkstemp()

        try:
            with os.fdopen(fd, "w") as f_temp:
                f_temp.write(tagged_parse)
            process = subprocess.run([str("bin64/TestGrid"), name], capture_output=True, env=self.env, cwd=str(self.brown_path))
        finally:
            os.remove(name)
        return process.stdout.decode("utf8")

#java -cp "opennlp-tools-1.4.3.jar:/home/mitarb/steen/Documents/external_projects/entity_grid/maxent/output/maxent-2.5.3.jar:trove4j-2.0.2.jar" opennlp.tools.lang.english.NameFinder -parse *.bin

ROOT_REPLACE_RE = re.compile(r"\(ROOT")
NEWLINE_RE = re.compile(r"\)\s*\(TOP")

def process_parse(text):
    text = ROOT_REPLACE_RE.sub("(TOP", text)
    text = " ".join(text.split())
    text = NEWLINE_RE.sub(")\n(TOP", text)

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=Path)
    parser.add_argument("--parser", dest="parser_path", default="browncoherence/lib/stanford-parser")
    parser.add_argument("--model", dest="model_path", default="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    parser.add_argument("--tagger", dest="open_nlp_path", default="browncoherence/lib/opennlp/")
    parser.add_argument("--tagger-models", dest="open_nlp_models", default="browncoherence/lib/opennlp_models")
    parser.add_argument("--brown-path", dest="brown_path", default="browncoherence")
    parser.add_argument("--brown-lib", dest="brown_lib_path", default=Path("browncoherence/lib/gsl-2.6/lib/"), type=Path)
    parser.add_argument("--parses-dir", dest="parses_dir", default=Path("outputs/stanford-parses"), type=Path)
    parser.add_argument("--grids-dir", dest="grids_dir", default=Path("outputs/entity-grids"), type=Path)
    args = parser.parse_args()

    parser = StanfordParser(
        args.parser_path,
        args.model_path
    )
    tagger = OpenNLPTagger(args.open_nlp_path, args.open_nlp_models)
    entity_grid_builder = EntityGridBuilder(args.brown_path, args.brown_lib_path)

    args.grids_dir.mkdir(exist_ok=True, parents=True)
    args.parses_dir.mkdir(exist_ok=True, parents=True)
    
    with open(args.infile) as f, open(args.grids_dir / args.infile.name, "w") as f_grids, open(args.parses_dir / args.infile.name, "w") as f_parses:
        reader = csv.reader(f)
        writer = csv.writer(f_grids, lineterminator='\n')
        parses_writer = csv.writer(f_parses, lineterminator='\n')

        index = next(reader)[:-1]
        writer.writerow(index + ["grid"])
        parses_writer.writerow(index + ["parse"])

        for line in tqdm.tqdm(reader):
            parse = process_parse(parser.parse(line[-1]))
            tagged = tagger.tag(parse)
            entity_grid = entity_grid_builder.create_grid(tagged)
            
            writer.writerow(line[:-1] + [entity_grid])
            parses_writer.writerow(line[:-1] + [tagged])



if __name__ == "__main__":
    main()


