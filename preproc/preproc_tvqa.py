import json
import pandas as pd
import pickle
import torch
from args import DATA_DIR

with open(f"{DATA_DIR}/TVQA/tv_subtitles.jsonl") as f:
    data = [json.loads(line) for line in f]
subs = {x["vid_name"]: x["sub"] for x in data}
pickle.dump(subs, open(f"{DATA_DIR}/TVQA/subtitles.pkl", "wb"))

feats = torch.load(f"{DATA_DIR}/TVQA/clipvitl14_features.pth")

splits = ["train", "val", "test_public", "test_release"]
for split in splits:
    with open(f"{DATA_DIR}/TVQA/tvqa_{split}.jsonl") as f:
        data = [json.loads(line) for line in f]
    video_id = [x["vid_name"] for x in data]
    start = [float(x["ts"].split("-")[0]) for x in data]
    end = [float(x["ts"].split("-")[1]) for x in data]
    a0 = [
        x["a0"].strip()[:-1] if x["a0"].strip()[-1] == "." else x["a0"].strip()
        for x in data
    ]
    a1 = [
        x["a1"].strip()[:-1] if x["a1"].strip()[-1] == "." else x["a1"].strip()
        for x in data
    ]
    a2 = [
        x["a2"].strip()[:-1] if x["a2"].strip()[-1] == "." else x["a2"].strip()
        for x in data
    ]
    a3 = [
        x["a3"].strip()[:-1] if x["a3"].strip()[-1] == "." else x["a3"].strip()
        for x in data
    ]
    a4 = [
        x["a4"].strip()[:-1] if x["a4"].strip()[-1] == "." else x["a4"].strip()
        for x in data
    ]
    question = [x["q"] for x in data]
    qid = [x["qid"] for x in data]
    if split not in ["test_public", "test_release"]:
        answer_id = [x["answer_idx"] for x in data]
        df = pd.DataFrame(
            {
                "qid": qid,
                "video_id": video_id,
                "start": start,
                "end": end,
                "question": question,
                "a0": a0,
                "a1": a1,
                "a2": a2,
                "a3": a3,
                "a4": a4,
                "answer_id": answer_id,
            },
            columns=[
                "qid",
                "video_id",
                "start",
                "end",
                "question",
                "a0",
                "a1",
                "a2",
                "a3",
                "a4",
                "answer_id",
            ],
        )
    else:
        df = pd.DataFrame(
            {
                "qid": qid,
                "video_id": video_id,
                "start": start,
                "end": end,
                "question": question,
                "a0": a0,
                "a1": a1,
                "a2": a2,
                "a3": a3,
                "a4": a4,
            },
            columns=[
                "qid",
                "video_id",
                "start",
                "end",
                "question",
                "a0",
                "a1",
                "a2",
                "a3",
                "a4",
            ],
        )
    print(len(df))
    print(len(df[df["video_id"].isin(feats)]))
    df.to_csv(f"{DATA_DIR}/TVQA/{split}.csv", index=False)
