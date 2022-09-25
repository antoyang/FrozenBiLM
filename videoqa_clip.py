import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import argparse
from util import dist
from torch.utils.data import DataLoader, DistributedSampler
from collections import namedtuple
from functools import reduce

from datasets import build_videoqa_dataset_clip, videoqa_collate_fn
import clip
from args import get_args_parser, MODEL_DIR
from util.metrics import MetricLogger


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    dataset_name,
    args,
    thresholds=[1, 10],
    split="test",
    type_map={0: "all"},
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"{split}:"
    res = {}

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device)
        text = batch_dict["text"]
        text = [item for sublist in text for item in sublist]
        encoded = clip.tokenize(text, truncate=True).to(device)
        text_features = (
            model.encode_text(encoded)
            .view(-1, len(video), video.shape[-1])
            .transpose(0, 1)
        )
        logits = torch.bmm(
            video[:, 0].float().unsqueeze(1), text_features.transpose(1, 2).float()
        ).squeeze(
            1
        )  # match visual features with text (ie all possible question-answer) features
        topk_aids = torch.topk(logits, max(thresholds), -1).indices

        answer_id, qids = batch_dict["answer_id"].to(device), batch_dict["qid"]
        types = batch_dict["type"]
        if "sub" in batch_dict:
            subs = batch_dict["sub"]
        else:
            subs = [0] * len(types)
        if dataset_name not in ["ivqa", "vqa"]:
            answer_id_expanded = answer_id.view(-1, 1).expand_as(topk_aids).to(device)
        elif dataset_name == "ivqa":
            answer_id = (answer_id / 2).clamp(max=1)
            answer_id_expanded = answer_id.to(device)
        elif dataset_name == "vqa":
            answer_id = (answer_id / 3).clamp(max=1)
            answer_id_expanded = answer_id.to(device)

        agreeings = {}
        for x in thresholds:
            if dataset_name not in ["ivqa", "vqa"]:
                agreeings[x] = topk_aids[:, :x] == answer_id_expanded[:, :x]
            else:
                predicted = F.one_hot(
                    topk_aids[:, :x], num_classes=answer_id_expanded.shape[-1]
                ).sum(1)
                agreeings[x] = (predicted * answer_id_expanded).max(1)[0]

        for i, (qid, gt, pred, type, sub) in enumerate(
            zip(qids, answer_id, topk_aids, types, subs)
        ):
            res[qid] = {
                "pred": pred.tolist(),
                "gt": gt.tolist() if dataset_name in ["ivqa", "vqa"] else gt.item(),
                "type": int(type),
                "sub": sub,
            }
            for x in thresholds:
                res[qid][f"acc{x}"] = agreeings[x][i].sum().detach().cpu().item()

        dico = {"acc": agreeings[1].sum() / len(qids)}
        dico_reduced = dist.reduce_dict(dico)
        acc_value = dico_reduced["acc"].item()
        metric_logger.update(acc=acc_value)

    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    assert len(results) == len(data_loader.dataset)
    out = {}
    for x in thresholds:
        out[f"acc{x}"] = sum(results[qid][f"acc{x}"] for qid in results) / len(results)
    if type_map is not None and len(type_map) > 1:
        acc_type = {
            type_map[i]: sum(
                results[qid][f"acc1"] for qid in results if results[qid]["type"] == i
            )
            / len([x for x in results.values() if x["type"] == i])
            for i in type_map
        }
    n_sub = len([x for x in results.values() if x["sub"]])
    if n_sub:
        acc_sub = (
            sum(results[qid][f"acc1"] for qid in results if results[qid]["sub"]) / n_sub
        )
    if dist.is_main_process():
        print(dataset_name)
        for x in thresholds:
            print(f"{split} acc{x}: {out[f'acc{x}']: .2%}")
        if type_map is not None and len(type_map) > 1:
            for x in acc_type:
                print(f"acc {x}: {acc_type[x]: .2%}")
            out.update(acc_type)
        if n_sub:
            print(f"acc sub: {acc_sub: .2%}; proportion {n_sub / len(results): .2%}")
            out["acc_sub"] = acc_sub

    return results, out


def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)

    if dist.is_main_process():
        if args.save_dir and not (os.path.isdir(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir), exist_ok=True)
        print(args)

    device = torch.device(args.device)

    # Fix seeds
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model

    nt = namedtuple(
        typename="data",
        field_names=[
            "dataset_name",
            "dataloader_test",
            "dataloader_val",
        ],
    )

    tuples = []
    if not args.eval:
        raise NotImplementedError
    for dset_name in args.combine_datasets_val:
        dataset_test = build_videoqa_dataset_clip(
            dset_name, "val" if (args.eval and not args.test) else "test", args
        )
        sampler_test = (
            DistributedSampler(dataset_test, shuffle=False)
            if args.distributed
            else torch.utils.data.SequentialSampler(dataset_test)
        )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=args.batch_size_val,
            sampler=sampler_test,
            collate_fn=videoqa_collate_fn,
            num_workers=args.num_workers,
        )

        dataset_val = build_videoqa_dataset_clip(dset_name, "val", args)
        sampler_val = (
            DistributedSampler(dataset_val, shuffle=False)
            if args.distributed
            else torch.utils.data.SequentialSampler(dataset_val)
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size_val,
            sampler=sampler_val,
            collate_fn=videoqa_collate_fn,
            num_workers=args.num_workers,
        )

        tuples.append(
            nt(
                dataset_name=dset_name,
                dataloader_test=dataloader_test,
                dataloader_val=dataloader_val,
            )
        )

    assert args.max_feats == 1  # clip
    model, _ = clip.load("ViT-L/14", download_root=MODEL_DIR, device=device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if dist.is_main_process():
        print("number of params:", n_parameters)

    # Load pretrained checkpoint
    if args.load:
        if dist.is_main_process():
            print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)

    for i, item in enumerate(tuples):
        results, out = evaluate(
            model=model,
            data_loader=item.dataloader_test,
            device=device,
            dataset_name=item.dataset_name,
            args=args,
            split="val" if (args.eval and not args.test) else "test",
            type_map=item.dataloader_test.dataset.type_map,
        )

        if args.save_dir and dist.is_main_process():
            json.dump(
                results,
                open(os.path.join(args.save_dir, item.dataset_name + ".json"), "w"),
            )
            json.dump(
                out,
                open(
                    os.path.join(args.save_dir, item.dataset_name + "summary.json"), "w"
                ),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
    args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
