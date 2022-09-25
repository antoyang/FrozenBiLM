import json
import pandas as pd
import pickle
import torch
from tqdm import tqdm
from args import DATA_DIR

id2vid = json.load(open(f"{DATA_DIR}/How2QA/how2_vid_mapping.json", "r"))

# convert jsonlines to pickle for the subtitles
with open(f"{DATA_DIR}/How2QA/subtitles.jsonl") as f:
    data = [json.loads(line) for line in f]
subs = {x["vid_name"]: x["sub"] for x in data}
pickle.dump(subs, open(f"{DATA_DIR}/How2QA/subtitles.pkl", "wb"))

# convert video features extracted as one per YouTube video to one per video clip
feats = torch.load(f"{DATA_DIR}/How2QA/clipvitl14.pth")
new_feats = {}
for x in tqdm(subs):
    if "_".join(x.split("_")[:-2]) in feats:
        new_feats[x] = feats["_".join(x.split("_")[:-2])][
            int(x.split("_")[-2]) : int(x.split("_")[-1]) + 1
        ].clone()
    elif x in feats:
        new_feats[x] = feats[x].clone()
    else:
        print(x)
print(len(new_feats), len(subs))
torch.save(new_feats, f"{DATA_DIR}/How2QA/clipvitl14_split.pth")

# get train
splits = ["train"]
for split in splits:
    with open(f"{DATA_DIR}/how2qa_{split}_release.jsonl") as f:
        data = [json.loads(line) for line in f]
    video_id = [id2vid.get(x["vid_name"], x["vid_name"]) for x in data]
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
    question = [x["q"] for x in data]
    qid = [x["qid"] for x in data]
    if split != "test_public":
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
            ],
        )
    print(len(df))
    print(len(df[df["video_id"].isin(new_feats)]))
    df.to_csv(f"{DATA_DIR}/How2QA/{split}.csv", index=False)

# get public val
df = pd.read_csv(f"{DATA_DIR}/How2QA/how2QA_val_release.csv")
df.columns = ["vid_id", "timesteps", "a1", "a2", "a3", "question", "a0"]
print(len(df))
count = {}


def process(df):
    ids, a0, a1, a2, a3, answer, question, starts, ends, qid = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, row in tqdm(df.iterrows()):
        start = int(float(row["timesteps"].split(":")[0][1:]))
        sixty = start // 60
        end = int(float(row["timesteps"].split(":")[1][:-1]))
        vid_id = row["vid_id"] + "_" + str(sixty * 60) + "_" + str((sixty + 1) * 60)
        rel_start = start - sixty * 60
        rel_end = end - sixty * 60
        ids.append(vid_id)
        starts.append(rel_start)
        ends.append(rel_end)
        a0.append(row["a0"].strip()[:-1] if row["a0"][-1] == "." else row["a0"].strip())
        a1.append(row["a1"].strip()[:-1] if row["a1"][-1] == "." else row["a1"].strip())
        a2.append(row["a2"].strip()[:-1] if row["a2"][-1] == "." else row["a2"].strip())
        a3.append(row["a3"].strip()[:-1] if row["a3"][-1] == "." else row["a3"].strip())
        answer.append(0)
        question.append(row["question"])
        qid.append(i)
    return question, answer, ids, a0, a1, a2, a3, starts, ends, qid


question, answer, ids, a0, a1, a2, a3, starts, ends, qid = process(df)
val_df = pd.DataFrame(
    {
        "qid": qid,
        "question": question,
        "answer_id": answer,
        "video_id": ids,
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "start": starts,
        "end": ends,
    },
    columns=[
        "qid",
        "question",
        "answer_id",
        "video_id",
        "a0",
        "a1",
        "a2",
        "a3",
        "start",
        "end",
    ],
)
print(len(val_df))
print(val_df)
val_df.to_csv(f"{DATA_DIR}/How2QA/public_val.csv", index=False)
