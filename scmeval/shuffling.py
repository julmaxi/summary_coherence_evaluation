from pathlib import Path
import random
import itertools as it

from collections import Counter, namedtuple
from typing import List, Tuple, Generator
import json

from nltk.tokenize import word_tokenize
import datasets
import argparse

import pandas as pd

def select_indices(l, indices):
    out = []

    for i in indices:
        out.append(l[i])

    return out

def make_permutations(sentences, max_perms, allow_repetition=False):
    if len(sentences) == 1:
        return

    if len(sentences) <= 6 and not allow_repetition:
        perm_generator = list(it.permutations(list(range(len(sentences)))))
        random.shuffle(perm_generator)
        perm_generator.remove(tuple(range(len(sentences))))
        perm_generator = perm_generator[:max_perms]
    else:
        used_orders = set()
        orig_order = tuple(range(len(sentences)))
        used_orders.add(orig_order)
        def gen_shuffle():
            new_order = None
            while new_order is None or new_order in used_orders:
                new_order = list(orig_order)
                random.shuffle(new_order)
                new_order = tuple(new_order)
            if not allow_repetition:
                used_orders.add(new_order)
            return new_order

        perm_generator = (gen_shuffle() for _ in range(max_perms))

    for order in perm_generator:
        yield select_indices(sentences, order)


def make_local_permutations(sentences, n_swaps, max_perms, allow_repetition=True):
    if len(sentences) < 3 * n_swaps:
        return
    if len(sentences) < 10:
        return
    
    used_orders = set()
    orig_order = tuple(range(len(sentences)))
    used_orders.add(orig_order)
    def gen_shuffle():
        new_order = None
        while new_order is None or new_order in used_orders:
            new_order = list(orig_order)
            # It is not actually clear how the permutations are supposed
            # to be distributed. In this algorithm we sample indices left
            # to right. However, this means late starts for the first
            # window are more likely than they should be if we were to draw uniformly
            # from all possible starts, since late starts allow for fewer permutation down
            # the line
            permutation_starts = []

            cursor = 0

            for idx in range(n_swaps):
                start = random.randint(cursor, len(sentences) - (n_swaps - idx - 1) * 3 - 3)
                cursor = start
                permutation_starts.append(cursor)
            
            for start in permutation_starts:
                window = new_order[start:start + 3]
                random.shuffle(window)
                new_order[start:start + 3] = window
            new_order = tuple(new_order)
        if not allow_repetition:
            used_orders.add(new_order)
        return new_order

    perm_generator = (gen_shuffle() for _ in range(max_perms))
    for order in perm_generator:
        yield select_indices(sentences, order)


class Instance:
    def __init__(
        self,
        original_sentences: List[str],
        permuted_instances: List[List[str]],
        doc_name: str,
        dataset_split: str
    ) -> None:
        self.original_sentences = original_sentences
        self.permuted_instances = permuted_instances
        self.doc_name = doc_name
        self.dataset_split = dataset_split


class UnifiedModelDataWriter:
    def __init__(self, out_path: Path, max_length=50):
        self.out_path = out_path
        self.docs_dir = self.out_path / "docs"
        self.train_docs_dir = self.docs_dir / "train"
        self.test_docs_dir = self.docs_dir / "test"

        self.train_files = []
        self.test_files = []

        self.vocab_counter = Counter()
        self.max_length = max_length

    def prepare(self):
        self.out_path.mkdir()
        self.train_docs_dir.mkdir(exist_ok=True, parents=True)
        self.test_docs_dir.mkdir(exist_ok=True, parents=True)
    
    def __call__(self, instance: Instance):
        assert instance.dataset_split in ["train", "test"]

        if len(instance.original_sentences) > self.max_length:
            return

        if instance.dataset_split == "train":
            self.vocab_counter.update([t.lower() for s in instance.original_sentences for t in word_tokenize(s)])
        
        for idx, perm in enumerate(instance.permuted_instances):
            if instance.dataset_split == "train":
                out_doc_dir = self.train_docs_dir
            else:
                out_doc_dir = self.test_docs_dir

            if instance.dataset_split == "train":
                self.train_files.append(instance.doc_name)
            else:
                self.test_files.append(instance.doc_name)
            
            with open(out_doc_dir / (instance.doc_name + f"_{idx + 1}"), "w") as f:
                json.dump(
                    [
                        [f'<bos> {" ".join(word_tokenize(s.lower()))} <eos>' for s in instance.original_sentences],
                        [f'<bos> {" ".join(word_tokenize(s.lower()))} <eos>' for s in perm]
                    ],
                    f
                )

    def finalize(self):
        vocab = ["<pad>", "<unk>", "<bos>", "<eos>"] + list(self.vocab_counter.keys())

        with open(self.out_path / "vocab", "w") as f:
            json.dump(vocab, f)

        (self.out_path / "train.flist").write_text("\n".join(self.train_files))
        (self.out_path / "test.flist").write_text("\n".join(self.test_files))


def sent_split_newlines(text: str) -> List[str]:
    sentences = text.split("\n")
    sentences = (s.strip() for s in sentences)
    return list(filter(lambda s: len(s) > 0, sentences))


class SpacySplitter:
    def __init__(self) -> None:
        import spacy
        self.nlp = spacy.load("en_core_web_sm")
    
    def __call__(self, text: str) -> List[str]:
        parsed_text = self.nlp(text)
        return [s for sent in parsed_text.sents if len(s := sent.text_with_ws.strip()) > 0]


def load_wsj_dataset(path: Path) -> Generator[Tuple[str, str], None, None]:
    for doc in path.glob("**/wsj*"):
        fname = doc.parts[-1]
        dirname = fname.split("_")[1][:2]
        
        text = doc.read_text(encoding="latin1")
        text = text.replace(".START", "")

        yield text, fname, int(dirname)


def dataset_def(s: str) -> Tuple[str, str, "str"]:
    if s.startswith("wsj:"):
        return ("wsj", s.split(":")[1], "text")
    else:
        return ("default", "cnn_dailymail", "highlights")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=dataset_def)
    parser.add_argument("-f", "--format", dest="format", default="unified", choices=["unified", "dataset", "csv-single"])
    parser.add_argument("-d", "--dest", dest="destination")
    parser.add_argument("-m", "--mode", dest="shuffle_mode", default="global")
    parser.add_argument("-n", "--samples", dest="num_samples", default=20, type=int)
    parser.add_argument("-s", "--split", dest="split_mode", default="line", choices=["spacy", "line"])
    parser.add_argument("-l", "--limit", dest="limit", default=None, type=int)

    args = parser.parse_args()

#    Shuffling is not deterministic due to parallelization
#    random.seed(args.seed)

    if args.split_mode == "line":
        sent_splitter = sent_split_newlines
    else:
        sent_splitter = SpacySplitter()

    if args.shuffle_mode == "local":
        args.num_samples = [int(s) for s in args.num_samples.split(":")]

    if args.dataset[0] == "default":
        dataset = datasets.load_dataset("cnn_dailymail", "3.0.0")
        dataset["train"] = dataset["test"]
        print("WARNING, using CNN/DM testset for evaluation")
        dataset = dataset.map(lambda x: {"doc_name": x["id"]})
    else:
        items = list(load_wsj_dataset(Path(args.dataset[1])))
        df_train = pd.DataFrame.from_records([{"text": text, "doc_name": fname} for text, fname, split_id in items if split_id < 14])
        df_test = pd.DataFrame.from_records([{"text": text, "doc_name": fname} for text, fname, split_id in items if split_id >= 14])
        dataset_train = datasets.Dataset.from_pandas(df_train)
        dataset_test = datasets.Dataset.from_pandas(df_test)
        dataset = datasets.dataset_dict.DatasetDict({"train": dataset_train, "test": dataset_test})
    
    def add_permutations_to_dataset(items):
        out = {"sentences": [], "permutations": []}
        old_columns = list(items.keys())
        for col in old_columns:
            out[col] = []
        for idx, item in enumerate(items[args.dataset[2]]):
            sentences = sent_splitter(item)

            if args.shuffle_mode == "global":
                perms = list(make_permutations(sentences, args.num_samples))
                if len(perms) < args.num_samples:
                    continue

            else:
                perms = []
                for window_idx, num_samples in enumerate(args.num_samples):
                    window_count = window_idx + 1
                    perms.extend(make_local_permutations(sentences, window_count, num_samples))
                
                if len(perms) < sum(args.num_samples):
                    continue

            out["sentences"].append(sentences)
            out["permutations"].append(perms)

            for col in old_columns:
                out[col].append(items[col][idx])
            
        return out

    dataset = dataset.map(add_permutations_to_dataset, batched=True).shuffle()

    if args.limit is not None:
        dataset["train"] = dataset["train"].train_test_split(train_size=1000)["train"]

    if args.format == "unified":
        writer = UnifiedModelDataWriter(Path(args.destination))
        writer.prepare()
        for elem in dataset["train"]:
            instance = Instance(elem["sentences"], elem["permutations"], elem["doc_name"], "train")
            writer(instance)
        for elem in dataset["test"]:
            instance = Instance(elem["sentences"], elem["permutations"], elem["doc_name"], "test")
            writer(instance)
        writer.finalize()
    elif args.format == "dataset":
        dataset.save_to_disk(args.destination)
    elif args.format == "csv-single":
        with open(args.destination, "w") as f:
            import csv
            writer = csv.writer(f, dialect="unix")
            writer.writerow(["id", "label", "text"])

            for elem in dataset["train"]:
                writer.writerow([elem["doc_name"], 1, "\n".join(elem["sentences"])])

                for perm in elem["permutations"]:
                    writer.writerow([elem["doc_name"], 0, "\n".join(perm)])


if __name__ == "__main__":
    main()
