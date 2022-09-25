import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import math
import argparse
from util import dist
from torch.utils.data import DataLoader, DistributedSampler
from collections import namedtuple
from functools import reduce

from datasets import build_videoqa_dataset_ar, videoqa_collate_fn_ar
from model import build_model, get_tokenizer
from main import get_args_parser
from util.misc import get_mask
from util.metrics import MetricLogger


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    tokenizer,
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

    a2id = data_loader.dataset.a2id
    # batch per answer length
    valid_tokids = {}
    valid_aids = {}
    for a, aid in a2id.items():
        tok = tokenizer(a, add_special_tokens=False)["input_ids"] + [
            tokenizer.eos_token_id
        ]
        if len(tok) not in valid_tokids:
            valid_tokids[len(tok)] = []
            valid_aids[len(tok)] = []
        valid_tokids[len(tok)].append(tok)
        valid_aids[len(tok)].append(aid)
    for l in valid_tokids:
        valid_tokids[l] = torch.tensor(valid_tokids[l], dtype=torch.long).to(device)

    if dist.is_main_process():
        print(
            len(a2id),
            sum(len(x) for y, x in valid_tokids.items() if y <= args.max_atokens),
        )
    range_alen = [i for i in range(1, args.max_atokens + 1) if i in valid_tokids]
    res = {}

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device)
        video_len = batch_dict["video_len"]
        video_mask = get_mask(video_len, video.size(1)).to(device)
        text = batch_dict["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=args.max_tokens,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        attention_mask = (
            torch.cat([video_mask, encoded["attention_mask"].to(device)], 1)
            if args.use_video
            else encoded["attention_mask"].to(device)
        )
        input_ids = encoded["input_ids"].to(device)
        bests = {}
        for alen in range_alen:
            # forward by batch_size answers
            n_ans = len(valid_tokids[alen])
            n_fwds = math.ceil(n_ans / args.batch_size_val)
            for n_fwd in range(n_fwds):
                cur_len = len(
                    valid_tokids[alen][
                        n_fwd * args.batch_size_val : (n_fwd + 1) * args.batch_size_val
                    ]
                )
                output = model.score(
                    video=video.repeat(cur_len, 1, 1),
                    input_ids=input_ids.repeat(cur_len, 1),
                    target_ids=valid_tokids[alen][
                        n_fwd * args.batch_size_val : (n_fwd + 1) * args.batch_size_val
                    ],
                    attention_mask=attention_mask.repeat(cur_len, 1),
                )  # V L
                output_pool = output.prod(-1)  # prod probas
                best = torch.max(output_pool, 0)
                score, pred_n = (
                    best.values.item(),
                    valid_aids[alen][n_fwd * args.batch_size_val + best.indices.item()],
                )
                bests[pred_n] = score
        pred = max(bests, key=bests.get)
        preds = torch.tensor([pred], dtype=torch.long).to(device)
        answer_id, qids = batch_dict["answer_id"].to(device), batch_dict["qid"]
        if dataset_name == "ivqa":
            answer_id = (answer_id / 2).clamp(max=1)
        types = batch_dict["type"]

        if dataset_name != "ivqa":
            agreeings = preds == answer_id
        else:
            predicted = F.one_hot(preds, num_classes=answer_id.shape[1])
            agreeings = (predicted * answer_id).max(1)[0]

        for i, (qid, gt, pred, type) in enumerate(zip(qids, answer_id, preds, types)):
            res[qid] = {
                "pred": pred.item(),
                "gt": gt.tolist() if dataset_name == "ivqa" else gt.item(),
                "type": type.item(),
            }
            res[qid][f"acc"] = agreeings[i].sum().detach().cpu().item()

        dico = {"acc": agreeings.sum() / len(qids)}
        dico_reduced = dist.reduce_dict(dico)
        acc_value = dico_reduced["acc"].item()
        metric_logger.update(acc=acc_value)

    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    assert len(results) == len(data_loader.dataset)
    out = {}
    out[f"acc"] = sum(results[qid][f"acc"] for qid in results) / len(results)
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
        print(f"{split} acc: {out[f'acc']: .2%}")
        if type_map is not None and len(type_map) > 1:
            for x in acc_type:
                print(f"acc {x}: {acc_type[x]: .2%}")
            out.update(acc_type)

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
    model = build_model(args)
    model.to(device)
    tokenizer = get_tokenizer(args)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
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
        dataset_test = build_videoqa_dataset_ar(
            dset_name,
            "val" if (args.eval and not args.test) else "test",
            args,
        )
        sampler_test = (
            DistributedSampler(dataset_test, shuffle=False)
            if args.distributed
            else torch.utils.data.SequentialSampler(dataset_test)
        )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=1,  # one forward per answer in the vocab => batch per different answers
            sampler=sampler_test,
            collate_fn=videoqa_collate_fn_ar,
            num_workers=args.num_workers,
        )

        dataset_val = build_videoqa_dataset_ar(dset_name, "val", args)
        sampler_val = (
            DistributedSampler(dataset_val, shuffle=False)
            if args.distributed
            else torch.utils.data.SequentialSampler(dataset_val)
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=1,  # one forward per answer in the vocab => batch per different answers
            sampler=sampler_val,
            collate_fn=videoqa_collate_fn_ar,
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
        results, out = evaluate(
            model=model,
            tokenizer=tokenizer,
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
