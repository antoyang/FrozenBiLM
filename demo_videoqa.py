import os
import torch
import numpy as np
import random
import json
import argparse

from model import build_model, get_tokenizer
from args import get_args_parser
from util.misc import get_mask
import ffmpeg
from extract.preprocessing import Preprocessing
import clip
from args import MODEL_DIR


@torch.no_grad()
def main(args):
    assert args.question_example
    assert args.video_example

    device = torch.device(args.device)

    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model
    print("building model")
    tokenizer = get_tokenizer(args)
    vocab = json.load(open(args.msrvtt_vocab_path, "r"))
    id2a = {y: x for x, y in vocab.items()}
    args.n_ans = len(vocab)
    model = build_model(args)
    model.to(device)
    model.eval()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # Load pretrained checkpoint
    assert args.load
    print("loading from", args.load)
    checkpoint = torch.load(args.load, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    # Init answer embedding module
    aid2tokid = torch.zeros(len(vocab), args.max_atokens).long()
    for a, aid in vocab.items():
        tok = torch.tensor(
            tokenizer(
                a,
                add_special_tokens=False,
                max_length=args.max_atokens,
                truncation=True,
                padding="max_length",
            )["input_ids"],
            dtype=torch.long,
        )
        aid2tokid[aid] = tok
    model.set_answer_embeddings(aid2tokid.to(device), freeze_last=args.freeze_last)

    # Load video
    print("loading visual backbone")
    video_path = args.video_example
    preprocess = Preprocessing()
    backbone, _ = clip.load("ViT-L/14", download_root=MODEL_DIR, device=device)
    backbone.eval()

    # Extract frames from video
    print("extracting visual features")
    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    num, denum = video_stream["avg_frame_rate"].split("/")
    frame_rate = int(num) / int(denum)
    if height >= width:
        h, w = int(height * 224 / width), 224
    else:
        h, w = 224, int(width * 224 / height)
    assert frame_rate >= 1

    cmd = ffmpeg.input(video_path).filter("fps", fps=1).filter("scale", w, h)
    x = int((w - 224) / 2.0)
    y = int((h - 224) / 2.0)
    cmd = cmd.crop(x, y, 224, 224)
    out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
        capture_stdout=True, quiet=True
    )

    h, w = 224, 224
    video = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
    video = torch.from_numpy(video.astype("float32"))
    video = video.permute(0, 3, 1, 2)
    video = video.squeeze()
    video = preprocess(video)
    video = backbone.encode_image(video.to(device))

    # Subsample or pad
    if len(video) >= args.max_feats:
        sampled = []
        for j in range(args.max_feats):
            sampled.append(video[(j * len(video)) // args.max_feats])
        video = torch.stack(sampled)
        video_len = args.max_feats
    else:
        video_len = len(video)
        video = torch.cat(
            [video, torch.zeros(args.max_feats - video_len, 768).to(device)], 0
        )
    video = video.unsqueeze(0).to(device)
    video_mask = get_mask(
        torch.tensor(video_len, dtype=torch.long).unsqueeze(0), video.size(1)
    ).to(device)
    print("visual features extracted")

    # Process question
    question = args.question_example.capitalize().strip()
    if question[-1] != "?":
        question = str(question) + "?"
    text = f"{args.prefix} Question: {question} Answer: {tokenizer.mask_token}{args.suffix}"
    encoded = tokenizer(
        [text],
        add_special_tokens=True,
        max_length=args.max_tokens,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    if not args.suffix:  # remove sep token if not using the suffix
        attention_mask[input_ids == tokenizer.sep_token_id] = 0
        input_ids[input_ids == tokenizer.sep_token_id] = tokenizer.pad_token_id
    print("encoded text")

    output = model(
        video=video,
        video_mask=video_mask,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = output["logits"]
    delay = args.max_feats if args.use_video else 0
    logits = logits[:, delay : encoded["input_ids"].size(1) + delay][
        encoded["input_ids"] == tokenizer.mask_token_id
    ]  # get the prediction on the mask token
    logits = logits.softmax(-1)
    topk = torch.topk(logits, 5, -1)
    topk_txt = [[id2a[x.item()] for x in y] for y in topk.indices.cpu()]
    topk_scores = [[f"{x:.2f}".format() for x in y] for y in topk.values.cpu()]
    topk_all = [
        [x + "(" + y + ")" for x, y in zip(a, b)] for a, b in zip(topk_txt, topk_scores)
    ]
    print(f"Top 5 answers and scores: {topk_all[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
    args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
