import torch
from typing import Tuple, List
import random


def get_mask(lengths, max_length):
    """Computes a batch of padding masks given batched lengths"""
    mask = 1 * (
        torch.arange(max_length).unsqueeze(1).to(lengths.device) < lengths
    ).transpose(0, 1)
    return mask


def mask_tokens(inputs, tokenizer, mlm_probability):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
    )
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def adjust_learning_rate(
    optimizer,
    curr_step: int,
    num_training_steps: int,
    args,
):
    num_warmup_steps: int = round(args.fraction_warmup_steps * num_training_steps)
    if args.schedule == "linear_with_warmup":
        if curr_step < num_warmup_steps:
            gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            gamma = max(
                0.0,
                float(num_training_steps - curr_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )
    else:  # constant LR
        gamma = 1

    optimizer.param_groups[0]["lr"] = args.lr * gamma
