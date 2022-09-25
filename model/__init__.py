from .bert import BertForMaskedLM
from .deberta import DebertaV2ForMaskedLM
from .gptneo import GPTNeoForCausalLM
from .gptj import GPTJForCausalLM
from transformers import (
    BertTokenizer,
    DebertaV2Tokenizer,
    DebertaV2Config,
    BertConfig,
    GPT2Tokenizer,
)


def build_model(args):
    if "deberta" in args.model_name:
        if args.scratch:
            config = DebertaV2Config.from_pretrained(
                pretrained_model_name_or_path=args.model_name, local_files_only=True
            )
            model = DebertaV2ForMaskedLM(
                features_dim=args.features_dim if args.use_video else 0,
                max_feats=args.max_feats,
                freeze_lm=args.freeze_lm,
                freeze_mlm=args.freeze_mlm,
                ft_ln=args.ft_ln,
                ds_factor_attn=args.ds_factor_attn,
                ds_factor_ff=args.ds_factor_ff,
                dropout=args.dropout,
                n_ans=args.n_ans,
                freeze_last=args.freeze_last,
                config=config,
            )
        else:
            model = DebertaV2ForMaskedLM.from_pretrained(
                features_dim=args.features_dim if args.use_video else 0,
                max_feats=args.max_feats,
                freeze_lm=args.freeze_lm,
                freeze_mlm=args.freeze_mlm,
                ft_ln=args.ft_ln,
                ds_factor_attn=args.ds_factor_attn,
                ds_factor_ff=args.ds_factor_ff,
                dropout=args.dropout,
                n_ans=args.n_ans,
                freeze_last=args.freeze_last,
                pretrained_model_name_or_path=args.model_name,
                local_files_only=True,
            )
    elif "bert" in args.model_name:
        assert (
            (not args.ds_factor_ff) and (not args.ds_factor_attn) and (not args.scratch)
        )
        model = BertForMaskedLM.from_pretrained(
            features_dim=args.features_dim if args.use_video else 0,
            max_feats=args.max_feats,
            freeze_lm=args.freeze_lm,
            freeze_mlm=args.freeze_mlm,
            ft_ln=args.ft_ln,
            n_ans=args.n_ans,
            freeze_last=args.freeze_last,
            pretrained_model_name_or_path=args.model_name,
            local_files_only=True,
        )
    elif "gpt-neo" in args.model_name:
        assert (
            (not args.ds_factor_ff) and (not args.ds_factor_attn) and (not args.scratch)
        )
        model = GPTNeoForCausalLM.from_pretrained(
            features_dim=args.features_dim if args.use_video else 0,
            max_feats=args.max_feats,
            freeze_lm=args.freeze_lm,
            freeze_mlm=args.freeze_mlm,
            ft_ln=args.ft_ln,
            pretrained_model_name_or_path=args.model_name,
            local_files_only=True,
        )
    elif "gpt-j" in args.model_name:
        assert (
            (not args.ds_factor_ff) and (not args.ds_factor_attn) and (not args.scratch)
        )
        model = GPTJForCausalLM.from_pretrained(
            features_dim=args.features_dim if args.use_video else 0,
            max_feats=args.max_feats,
            freeze_lm=args.freeze_lm,
            freeze_mlm=args.freeze_mlm,
            ft_ln=args.ft_ln,
            pretrained_model_name_or_path=args.model_name,
            local_files_only=True,
        )
    else:
        raise NotImplementedError
    return model


def get_tokenizer(args):
    if "deberta" in args.model_name:
        tokenizer = DebertaV2Tokenizer.from_pretrained(
            args.model_name, local_files_only=True
        )
    elif "bert" in args.model_name:
        tokenizer = BertTokenizer.from_pretrained(
            args.model_name, local_files_only=True
        )
    elif "gpt-neo" in args.model_name or "gpt-j" in args.model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(
            args.model_name, local_files_only=True
        )
        tokenizer.pad_token_id = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
    else:
        raise NotImplementedError
    return tokenizer
