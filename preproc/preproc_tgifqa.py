import pandas as pd
import os
import collections
import json
import torch
from args import DATA_DIR

os.chdir(f"{DATA_DIR}/TGIF-QA")

train_frameqa = pd.read_csv(
    "dataset/Train_{}_question.csv".format("frameqa"), delimiter="\t"
)
test_frameqa = pd.read_csv(
    "dataset/Test_{}_question.csv".format("frameqa"), delimiter="\t"
)

# get vocabulary


def get_vocabulary():
    train_counter = collections.Counter(train_frameqa["answer"].values)
    most_common = train_counter.most_common(1000)
    vocab = {}
    for i, x in enumerate(most_common):
        vocab[x[0]] = i

    with open("vocab.json", "w") as outfile:
        json.dump(vocab, outfile)
    return vocab


# preprocess dataframes


def restrict_df(
    vocabulary,
    train_frameqa,
    test_frameqa,
):
    extracted = [x for x in torch.load(f"clipvitl14.pth")]
    print(len(train_frameqa))
    train_frameqa["video_id"] = train_frameqa["gif_name"]
    train_frameqa = train_frameqa[
        train_frameqa["gif_name"].isin(extracted)
    ]  # only keep training samples for which visual features are available
    print(len(train_frameqa))
    test_frameqa["video_id"] = test_frameqa["gif_name"]
    print(len(test_frameqa))
    train_frameqa = train_frameqa[train_frameqa["answer"].isin(vocabulary)]
    print(len(train_frameqa))
    train_frameqa.to_csv("dataset/train_{}.csv".format("frameqa"), index=False)
    test_frameqa.to_csv("dataset/test_{}.csv".format("frameqa"), index=False)
    return (
        train_frameqa,
        test_frameqa,
    )


vocabulary = get_vocabulary()
(train_frameqa, test_frameqa,) = restrict_df(
    vocabulary,
    train_frameqa,
    test_frameqa,
)
