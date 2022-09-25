import torch as th
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class VideoText_Dataset(Dataset):
    def __init__(self, csv_path, features_path, max_feats=10, features_dim=768):
        self.data = pd.read_csv(csv_path)
        self.features = features_path
        self.max_feats = max_feats
        self.features_dim = features_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data["text"].values[idx]

        video_id = self.data["video_id"].values[idx]
        try:
            video = th.from_numpy(
                np.load(os.path.join(self.features, str(video_id) + ".mp4.npy"))
            ).float()
            if len(video) > self.max_feats:
                sampled = []
                for j in range(self.max_feats):
                    sampled.append(video[(j * len(video)) // self.max_feats])
                video = th.stack(sampled)
                video_len = self.max_feats
            elif len(video) < self.max_feats:
                video_len = len(video)
                video = th.cat(
                    [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
                )
            else:
                video_len = self.max_feats
        except:  # missing video or corrupted feature file
            video = th.zeros(self.max_feats, self.features_dim)
            video_len = 0

        return {"video": video, "video_len": video_len, "text": text}


def videotext_collate_fn(batch):
    bs = len(batch)
    video = th.stack([batch[i]["video"] for i in range(bs)])
    video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
    text = [batch[i]["text"] for i in range(bs)]

    return {
        "video": video,
        "video_len": video_len,
        "text": text,
    }


def build_videotext_dataset(split, args):
    if split == "train":
        csv_path = args.webvid_train_csv_path
    elif split == "val":
        csv_path = args.webvid_val_csv_path
    else:
        raise NotImplementedError
    return VideoText_Dataset(
        csv_path=csv_path,
        features_path=args.webvid_features_path,
        max_feats=args.max_feats,
        features_dim=args.features_dim,
    )
