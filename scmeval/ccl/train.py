import random
import itertools as it
import argparse

import torch
import pandas as pd

import transformers
import datasets

import numpy as np


def concat_sentences(sentences, order):
    new_sent_order = []
    for idx in order:
        new_sent_order.append(sentences[idx])

    return "\n".join(new_sent_order)


def transpose_dict(list_of_dicts):
    keys = list(list_of_dicts[0].keys())

    output = {}
    for key in keys:
        output[key] = []

    for d in list_of_dicts:
        for key in keys:
            output[key].append(d[key])

    return output


def take(i, n):
    for _ in range(n):
        try:
            yield next(i)
        except StopIteration:
            return


def shuffle_item(items, n_shuffle=20):
    output = []
    col_name = "highlights"

    if col_name not in items:
        col_name = "summary"
    for idx in range(len(items[col_name])):
        text = items[col_name][idx]
        sentences = text.split("\n")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

        #indices = list(range(len(sentences)))
        #random.shuffle(indices)

        output.append({
            "text": "\n".join(sentences),
            "labels": 0
        })

        if len(sentences) == 1:
            continue

        if len(sentences) <= 6:
            perm_generator = list(it.permutations(list(range(len(sentences)))))
            random.shuffle(perm_generator)
            perm_generator.remove(tuple(range(len(sentences))))
            perm_generator = perm_generator[:n_shuffle]
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
                used_orders.add(new_order)
                return new_order
            
            perm_generator = (gen_shuffle() for _ in range(n_shuffle))


        for perm in perm_generator:
            shuffle_text = concat_sentences(sentences, perm)

            output.append({
                "text": shuffle_text,
                "labels": 1
            })

    return transpose_dict(output)


def parse_args():
    parser = argparse.ArgumentParser()

    def parse_dataset(s):
        parts = s.split(":", 1)

        return (parts[0], tuple(parts[1].split(":")))

    parser.add_argument("-d", dest="dataset", default=("rep", ("cnn_dailymail", "3.0.0")), type=parse_dataset)
    parser.add_argument("-t", dest="test_dataset", default=None, type=parse_dataset)
    parser.add_argument("-m", dest="base_model", default="roberta-large")

    args = parser.parse_args()
    return args


def load_dataset_from_spec(spec):
    type_, args = spec

    if type_ == "rep":
        return datasets.load_dataset(*args)
    elif type_ == "csv":
        return datasets.load_dataset("csv", data_files=args)
    else:
        raise ValueError(f"Invalid file type {type_}")


def main():
    args = parse_args()

    dataset = load_dataset_from_spec(args.dataset)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

    def tokenize(item):
        return tokenizer.batch_encode_plus(item["text"], padding="max_length", truncation=True, max_length=512)

    if not args.test_dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1)
    else:
        test_dataset = load_dataset_from_spec(args.test_dataset)
        dataset = datasets.DatasetDict({"train": dataset["train"], "test": test_dataset["train"]})

    to_remove = list(dataset["train"].column_names)
    dataset = dataset.map(shuffle_item, batched=True, remove_columns=to_remove, batch_size=10000)
    dataset = dataset.map(tokenize, batched=True)
    name = "trainer"
    if "wsj" in args.dataset[1][0]:
        name += "_wsj"
    else:
        name += "_cnndm"

    name += "_" + args.base_model

    print("Trainer:", name)
    training_args = transformers.TrainingArguments(name, per_device_train_batch_size=8, per_device_eval_batch_size=1, evaluation_strategy="epoch", learning_rate=2e-6, save_strategy="epoch", num_train_epochs=6)

    metric = datasets.load_metric("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        print("Preds:", predictions)
        return metric.compute(predictions=predictions, references=labels)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()
