import pandas as pd
import collections
import json
from args import DATA_DIR

splits = ["training", "val", "test"]
for split in splits:
    data = pd.read_csv(f"{DATA_DIR}/LSMDC/LSMDC16_annos_{split}_FIB.csv", sep="\t")
    data.columns = [
        "video_id",
        "start_aligned",
        "end_aligned",
        "start_extracted",
        "end_extracted",
        "sentence",
        "question",
        "answer",
    ]
    df = pd.DataFrame(
        {
            "video_id": list(data["video_id"]),
            "question": list(data["question"]),
            "answer": list(data["answer"]),
        },
        columns=["video_id", "question", "answer"],
    )
    if split == "training":  # construct vocabulary
        answers = collections.Counter(data["answer"]).most_common(1000)
        vocab = {x[0]: i for i, x in enumerate(answers)}
        print(len(df))
        df = df[df["answer"].isin(vocab)]
        json.dump(vocab, open(f"{DATA_DIR}/LSMDC/vocab.json", "w"))
    print(len(df))
    df = df[
        df["question"].str.contains("_____")
    ]  # remove some samples in the val set that do not contain blank
    print(len(df))
    df.to_csv(f"{DATA_DIR}/LSMDC/{split}.csv", index=False)
