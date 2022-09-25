import numpy as np
import argparse
import os
import torch
from tqdm import tqdm
import pandas as pd
from args import DATA_DIR

parser = argparse.ArgumentParser(description="Feature merger")

parser.add_argument("--folder", type=str, required=True, help="folder of features")
parser.add_argument(
    "--output_path", type=str, required=True, help="output path for features"
)
parser.add_argument(
    "--dataset",
    type=str,
    help="dataset",
    required=True,
    choices=[
        "lsmdc",
        "ivqa",
        "msrvtt",
        "msvd",
        "activitynet",
        "tgif",
        "how2qa",
        "tvqa",
    ],
)
parser.add_argument(
    "--pad",
    type=int,
    help="set as diff of 0 to trunc and pad up to a certain nb of seconds",
    default=0,
)

args = parser.parse_args()
files = os.listdir(args.folder)
files = [x for x in files if x[-4:] == ".npy"]

# Get mapping from feature file name to dataset video_id
if args.dataset == "msrvtt":
    mapping = {x: int(x.split(".")[0][5:]) for x in files}

elif args.dataset == "msvd":
    f = open(os.path.join(DATA_DIR, "MSVD-QA", "youtube_mapping.txt"))
    mapping = {}
    for line in f.readlines():
        l = line.split(" ")
        idx = l[1].split("\n")[0][3:]
        mapping[l[0] + ".avi.npy"] = int(idx)

elif args.dataset in ["ivqa", "activitynet"]:
    mapping = {x: x[:11] for x in files}

elif args.dataset in ["lsmdc", "tgif", "how2qa"]:
    mapping = {x: x[:-8] for x in files}

elif args.dataset == "tvqa":
    mapping = {}
    for i, row in pd.read_csv(os.path.join(DATA_DIR, "TVQA/feat_mapping.csv")):
        show = row["video_path"].split("/")[-2]
        if show == "bbt":
            prefix = ""
        else:
            prefix = show + "_"
        mapping[row["video_path"].split("/")[-1] + ".npy"] = (
            prefix + (row["video_path"].split("/")[-1])[:-4]
        )

else:
    raise NotImplementedError

features = {}
for i in tqdm(range(len(files))):
    x = files[i]
    feat = torch.from_numpy(np.load(os.path.join(args.folder, x)))
    if args.pad and len(feat) < args.pad:
        feat = torch.cat([feat, torch.zeros(args.pad - len(feat), feat.shape[1])])
    elif args.pad:
        feat = feat[: args.pad]
    features[mapping[x]] = feat.half()

torch.save(features, args.output_path)
