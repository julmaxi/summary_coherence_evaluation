import sys
import csv
from pathlib import Path

import pandas as pd

import torch
import transformers
import tqdm


def main():
    model = transformers.AutoModelForSequenceClassification.from_pretrained(sys.argv[1])
    tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-large")

    summaries = pd.read_csv(sys.argv[2])

    columns = list(summaries.columns)[:-1]
    summaries.set_index(columns, inplace=True)
    summaries = summaries[summaries.columns[0]]

    model.cuda()

    with torch.no_grad():
        values = []
        for summary in tqdm.tqdm(summaries):
            tokenization_info = tokenizer([summary], padding="max_length", truncation=True, max_length=512).convert_to_tensors("pt")

            tokenization_info.to("cuda")

            out = model(output_hidden_states=True, **tokenization_info)
            value = torch.nn.functional.softmax(out.logits, dim=-1)[0,0].item()
            values.append(value)

    pd.DataFrame.from_records({"simple_coherence_score": values}, index=summaries.index).to_csv(sys.argv[3])


if __name__ == "__main__":
    main()
