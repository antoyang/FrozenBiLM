import os
import torch
import numpy as np
import random
import json
import argparse
from util import dist
from torch.utils.data import DataLoader, DistributedSampler
from collections import namedtuple
from functools import reduce

from datasets import build_mc_dataset, mc_collate_fn
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
        logits_list = []
        for aid in range(len(text)):
            encoded = clip.tokenize(text[aid], truncate=True).to(device)
            text_features = model.encode_text(encoded)
            logits = (
                video[:, 0].float() @ (text_features.t().float())
            ).diag()  # match visual and text features
            logits_list.append(logits)
        logits = torch.stack(logits_list, 1)
        if logits.shape[1] == 1:
            preds = logits.round().long().squeeze(1)
        else:
            preds = logits.max(1).indices
        qids = batch_dict["qid"]
        types = batch_dict["type"]
        if batch_dict["answer_id"][0].item() != -1:
            answer_id = batch_dict["answer_id"].to(device)
            agreeings = preds == answer_id

            for i, (qid, gt, pred, type) in enumerate(
                zip(qids, answer_id, preds, types)
            ):
                res[qid] = (
                    {
                        "pred": pred.cpu().detach().item(),
                        "gt": gt.cpu().detach().item(),
                        "type": int(type),
                    }
                    if type_map is not None and len(type_map) > 1
                    else {
                        "pred": pred.cpu().detach().item(),
                        "gt": gt.cpu().detach().item(),
                    }
                )
                res[qid][f"acc"] = agreeings[i].cpu().detach().item()

            dico = {"acc": agreeings.sum() / len(qids)}
            dico_reduced = dist.reduce_dict(dico)
            acc_value = dico_reduced["acc"].item()
            metric_logger.update(acc=acc_value)
        else:
            for i, (qid, pred, type) in enumerate(zip(qids, preds, types)):
                res[str(qid)] = int(pred.cpu().detach().item())

    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    assert len(results) == len(data_loader.dataset)
    if isinstance(next(iter(results.values())), dict):
        acc = sum(int(results[qid][f"acc"]) for qid in results) / len(results)
        if type_map is not None and len(type_map) > 1:
            acc_type = {
                type_map[i]: sum(
                    results[qid][f"acc"] for qid in results if results[qid]["type"] == i
                )
                / len([x for x in results.values() if x["type"] == i])
                for i in type_map
            }
        if dist.is_main_process():
            print(dataset_name)
            print(f"{split} acc: {acc: .2%}")
            if type_map is not None and len(type_map) > 1:
                for x in acc_type:
                    print(f"acc {x}: {acc_type[x]: .2%}")

        return results, acc
    else:
        return results, 0


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
    assert args.max_feats == 1  # clip
    model, _ = clip.load("ViT-L/14", download_root=MODEL_DIR, device=device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if dist.is_main_process():
        print("number of params:", n_parameters)

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
        dataset_test = build_mc_dataset(
            dset_name,
            "val" if (args.eval and not args.test) else "test",
            args,
            None,
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
            collate_fn=mc_collate_fn,
            num_workers=args.num_workers,
        )

        dataset_val = build_mc_dataset(dset_name, "val", args, None)
        sampler_val = (
            DistributedSampler(dataset_val, shuffle=False)
            if args.distributed
            else torch.utils.data.SequentialSampler(dataset_val)
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size_val,
            sampler=sampler_val,
            collate_fn=mc_collate_fn,
            num_workers=args.num_workers,
        )

        tuples.append(
            nt(
                dataset_name=dset_name,
                dataloader_test=dataloader_test,
                dataloader_val=dataloader_val,
            )
        )

    # Load pretrained checkpoint
    if args.load:
        if dist.is_main_process():
            print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)

    for i, item in enumerate(tuples):
        results, acc = evaluate(
            model=model,
            data_loader=item.dataloader_test,
            device=device,
            dataset_name=item.dataset_name,
            args=args,
            type_map=item.dataloader_test.dataset.type_map,
            split="val" if (args.eval and not args.test) else "test",
        )

        if args.save_dir and dist.is_main_process():
            json.dump(
                results,
                open(
                    os.path.join(
                        args.save_dir,
                        item.dataset_name + "_val.json"
                        if (args.eval and not args.test)
                        else item.dataset_name + "_test.json",
                    ),
                    "w",
                ),
            )
            json.dump(
                {"acc": float(acc)},
                open(
                    os.path.join(
                        args.save_dir,
                        item.dataset_name + "acc_val.json"
                        if (args.eval and not args.test)
                        else item.dataset_name + "acc_test.json",
                    ),
                    "w",
                ),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
    args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
