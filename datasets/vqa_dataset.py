import torch as th
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import json
import pickle


class VQA_Dataset(Dataset):
    def __init__(
        self,
        pkl_path,
        features_path,
        max_feats=10,
        features_dim=768,
        vocab_path=None,
        train=False,
        prefix="",
        suffix="",
        tokenizer=None,
        type_map=None,
    ):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.features = th.load(features_path)
        self.max_feats = max_feats
        self.features_dim = features_dim
        self.a2id = json.load(open(vocab_path, "r"))
        print(len(data))
        self.data = []
        for idx in range(len(self.data)):
            answer = data[idx]["answer"]
            ok = False
            for a, soft_score in answer:
                if a in self.a2id and soft_score >= 3:
                    ok = True
            if ok:
                self.data.append(data[idx])
        print(len(self.data))
        self.train = train
        self.prefix = prefix
        self.suffix = suffix
        self.mask = tokenizer.mask_token
        self.type_map = type_map

    def __len__(self):
        return len(self.data)

    def _get_text(self, question, mask):
        return f"{self.prefix} Question: {question} Answer: {mask}{self.suffix}".strip()

    def __getitem__(self, idx):
        # get question
        question = self.data[idx]["question"].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
        type = self.data[idx]["type"]

        # get answer
        answer = self.data[idx]["answer"]
        answer_id = th.zeros(len(self.a2id))
        for a, soft_score in answer:
            if a in self.a2id:
                answer_id[self.a2id[a]] = soft_score
        final = []
        for x in answer:
            if x[1] >= 3:
                final.extend([x[0]] * 3)
            else:
                final.append(x[0] * x[1])
        answer = final

        # get text
        text = self._get_text(question, self.mask)

        # get features
        image_id = self.data[idx]["image_id"]
        video = (
            self.features[image_id].float().unsqueeze(0).repeat(self.max_feats, 1)
        )  # repeat the same frames max_feats time
        video_len = self.max_feats  # 1

        return {
            "video": video,
            "video_len": video_len,
            "text": text,
            "qid": idx,
            "answer_id": answer_id,
            "answer": answer,
            "type": type,
        }


def build_vqa_dataset(split, args, tokenizer):
    if split == "train":
        pkl_path = args.vqa_train_pkl_path
    elif split in [
        "val",
        "test",
    ]:  # test is reserved for leaderboards
        pkl_path = args.vqa_val_pkl_path
    else:
        raise NotImplementedError
    return VQA_Dataset(
        pkl_path=pkl_path,
        features_path=args.vqa_features_path,
        max_feats=args.max_feats,
        features_dim=args.features_dim,
        vocab_path=args.vqa_vocab_path,
        train=split == "train",
        prefix=args.prefix,
        suffix=args.suffix,
        tokenizer=tokenizer,
        type_map={0: "yesno", 1: "number", 2: "other"},
    )
