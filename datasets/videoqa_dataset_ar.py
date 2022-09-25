import torch as th
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import pandas as pd
import collections
import json
import pickle


class VideoQA_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        features_path,
        max_feats=10,
        features_dim=768,
        vocab_path=None,
        train=False,
        prefix="",
        fib=False,
        type_map=None,
        use_context=False,
        subtitles_path=None,
    ):
        self.data = pd.read_csv(csv_path)
        self.features = th.load(features_path)
        self.max_feats = max_feats
        self.features_dim = features_dim
        self.a2id = json.load(open(vocab_path, "r"))
        assert not train  # only implemented for eval
        self.prefix = prefix
        self.fib = fib
        self.type_map = type_map
        self.use_context = use_context
        if subtitles_path:
            self.subs = pickle.load(open(subtitles_path, "rb"))
        else:
            self.subs = None

    def __len__(self):
        return len(self.data)

    def _get_text(self, question, sub):
        text = (
            f"{self.prefix} Question: {question} Answer: "
            if not self.fib
            else f"{self.prefix} {question} Fill the blank: "
        )
        text = text.strip()
        if sub:
            text = f"Subtitles: {sub} " + text
        return text

    def _get_video(self, video_id):
        if video_id not in self.features:
            print(video_id)
            video = th.zeros(1, self.features_dim)
        else:
            video = self.features[video_id].float()
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

        return video, video_len

    def __getitem__(self, idx):
        # get question
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
        type = 0
        if "type" in self.data:
            type = self.data["type"].values[idx]

        # get answer
        if "answer" in self.data:
            answer = self.data["answer"].values[idx]
            answer_id = self.a2id.get(answer, -1)
        else:  # iVQA
            answer_list = [
                self.data["answer1"].values[idx],
                self.data["answer2"].values[idx],
                self.data["answer3"].values[idx],
                self.data["answer4"].values[idx],
                self.data["answer5"].values[idx],
            ]
            answer = collections.Counter(answer_list)
            answer_id = th.zeros(len(self.a2id))
            for x in answer:
                if x in self.a2id:
                    answer_id[self.a2id[x]] = answer[x]
            final = []
            for x in answer:
                if answer[x] >= 2:
                    final.extend([x] * 2)
                else:
                    final.append(x)
            answer = final

        video_id = self.data["video_id"].values[idx]

        # get subtitles
        sub = ""
        if self.subs is not None and video_id in self.subs:
            sub = self.subs[video_id]
        if not self.use_context:
            sub = ""

        # get pattern
        text = self._get_text(question, sub)

        # get features
        video, video_len = self._get_video(video_id)

        return {
            "video": video,
            "video_len": video_len,
            "text": text,
            "qid": idx,
            "answer": answer,
            "type": type,
            "answer_id": answer_id,
        }


def videoqa_collate_fn_ar(batch):
    bs = len(batch)
    video = th.stack([batch[i]["video"] for i in range(bs)])
    video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
    text = (
        [batch[i]["text"] for i in range(bs)]
        if isinstance(batch[0]["text"], str)
        else [
            [batch[i]["text"][j] for i in range(bs)]
            for j in range(len(batch[0]["text"]))
        ]
    )
    qid = [batch[i]["qid"] for i in range(bs)]
    answer = [batch[i]["answer"] for i in range(bs)]
    type = [batch[i]["type"] for i in range(bs)]
    answer_id = default_collate([batch[i]["answer_id"] for i in range(bs)])

    return {
        "video": video,
        "video_len": video_len,
        "text": text,
        "qid": qid,
        "answer": answer,
        "type": type,
        "answer_id": answer_id,
    }


def build_videoqa_dataset_ar(dataset_name, split, args):
    if dataset_name == "msvd":
        if split == "train":
            csv_path = args.msvd_train_csv_path
        elif split == "val":
            csv_path = args.msvd_val_csv_path
        elif split == "test":
            csv_path = args.msvd_test_csv_path
        else:
            raise NotImplementedError
        features_path = args.msvd_features_path
        vocab_path = args.msvd_vocab_path
        type_map = {0: "what", 1: "how", 2: "color", 3: "where", 4: "who", 5: "when"}
        subtitles_path = args.msvd_subtitles_path
    elif dataset_name == "msrvtt":
        if split == "train":
            csv_path = args.msrvtt_train_csv_path
        elif split == "val":
            csv_path = args.msrvtt_val_csv_path
        elif split == "test":
            csv_path = args.msrvtt_test_csv_path
        else:
            raise NotImplementedError
        features_path = args.msrvtt_features_path
        vocab_path = args.msrvtt_vocab_path
        type_map = {0: "what", 1: "how", 2: "color", 3: "where", 4: "who", 5: "when"}
        subtitles_path = args.msrvtt_subtitles_path
    elif dataset_name == "activitynet":
        if split == "train":
            csv_path = args.activitynet_train_csv_path
        elif split == "val":
            csv_path = args.activitynet_val_csv_path
        elif split == "test":
            csv_path = args.activitynet_test_csv_path
        else:
            raise NotImplementedError
        features_path = args.activitynet_features_path
        vocab_path = args.activitynet_vocab_path
        type_map = {
            0: "motion",
            1: "spatial",
            2: "temporal",
            3: "yesno",
            4: "color",
            5: "object",
            6: "location",
            7: "number",
            8: "other",
        }
        subtitles_path = args.activitynet_subtitles_path
    elif dataset_name == "ivqa":
        if split == "train":
            csv_path = args.ivqa_train_csv_path
        elif split == "val":
            csv_path = args.ivqa_val_csv_path
        elif split == "test":
            csv_path = args.ivqa_test_csv_path
        else:
            raise NotImplementedError
        features_path = args.ivqa_features_path
        vocab_path = args.ivqa_vocab_path
        type_map = None
        subtitles_path = args.ivqa_subtitles_path
    elif dataset_name == "tgif":
        if split == "train":
            csv_path = args.tgif_frameqa_train_csv_path
        elif split == "val":
            csv_path = args.tgif_frameqa_test_csv_path  # no val set in TGIF
        elif split == "test":
            csv_path = args.tgif_frameqa_test_csv_path
        else:
            raise NotImplementedError
        features_path = args.tgif_features_path
        vocab_path = args.tgif_vocab_path
        type_map = {0: "what", 1: "how", 2: "color", 3: "where"}
        subtitles_path = None
    elif dataset_name == "lsmdc":
        if split == "train":
            csv_path = args.lsmdc_train_csv_path
        elif split == "val":
            csv_path = args.lsmdc_val_csv_path
        elif split == "test":
            csv_path = args.lsmdc_test_csv_path
        features_path = args.lsmdc_features_path
        vocab_path = args.lsmdc_vocab_path
        type_map = None
        subtitles_path = args.lsmdc_subtitles_path

    if dataset_name in ["msvd", "msrvtt", "ivqa", "activitynet", "tgif", "lsmdc"]:
        return VideoQA_Dataset(
            csv_path=csv_path,
            features_path=features_path,
            max_feats=args.max_feats,
            features_dim=args.features_dim,
            vocab_path=vocab_path,
            train=split == "train",
            prefix=args.prefix,
            type_map=type_map,
            subtitles_path=subtitles_path,
            use_context=(args.use_context and dataset_name != "tgif"),
            fib=(dataset_name == "lsmdc"),
        )
    else:
        raise NotImplementedError
