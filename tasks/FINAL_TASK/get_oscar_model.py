# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import logging
import os

import torch

from model_oscar import ImageBertForSequenceClassificationwithAction, PreTrainOscar
from oscar.transformers_src.pytorch_transformers import BertConfig, BertTokenizer
from utils_data import load_detector_classes

logger = logging.getLogger(__name__)

MODEL_CLASS = {
    "ImageBertForSequenceClassificationwithAction": (
        BertConfig,
        ImageBertForSequenceClassificationwithAction,
        BertTokenizer,
    ),
    "PreTrainOscar": (
        BertConfig,
        PreTrainOscar,
        BertTokenizer,
    ),
}


special_tokens_dict = {
    "ques_token": "[QUES]",
    "ans_token": "[ANS]",
    "tar_token": "[TAR]",
}


def load_oscar_model(
    args,
    model_name,
    add_new_extra_embeds=True,
    finetuned=False,
):
    # Make sure only the first process in distributed training will download model & vocab
    if args.local_rank not in [-2, -1, 0]:
        torch.distributed.barrier()

    config_class, model_class, tokenizer_class = MODEL_CLASS[model_name]
    config = config_class.from_pretrained(args.model_name_or_path)

    if config is None:

        logger.info("No config exists at this path!!!!")
        tmp_root_folder = "srv/oscar_pretrained_models/base-vg-labels/ep_107_1192087"
        config_path = os.path.join(tmp_root_folder, "config.json")
        logger.info(f"Loading config from {config_path}")

        config = BertConfig.from_pretrained(config_path)

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
    )

    tokenizer.cls_token_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    if add_new_extra_embeds and not finetuned:
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(
            f"Added {num_added_toks} tokens {' '.join(special_tokens_dict.values())} to Tokenizer"
        )

    # config.img_feature_type # TODO: make sure this is not set to ""img_dis_code" or something. Check modeling_bert.py
    config.img_feature_dim = args.img_feature_dim
    config.hidden_dropout_prob = args.drop_out
    config.classifier = "linear"
    config.loss_type = "CrossEntropy"
    config.cls_hidden_scale = 2

    config.action_space = args.action_space

    config.detector_classes = len(load_detector_classes())

    if args.no_pretrained_model:
        model = model_class(config)
        logger.info(f"NO PRETRAINED MODEL LOADED !!!!!!")
    else:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

        logger.info(f"Loaded pretrained model {args.model_name_or_path}")

    # Make sure only the first process in distributed training will download model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()

    if add_new_extra_embeds and not finetuned:
        embedding_size_dict = {
            # adding [TAR], [QUES], [ANS]
            "word_embeddings": config.vocab_size + 3,
            "position_embeddings": args.max_seq_length,
            # token type for Target, Question, Answer, Region Label, Action Stream (1 already present, so +4)
            "token_type_embeddings": config.type_vocab_size + 4,
        }
        # config.vocab_size = config.vocab_size + 3
        model.resize_embeddings(embedding_size_dict)

    return model, tokenizer, config


def finetuned_model(args):
    pass
