# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import torch
import logging
from oscar.modeling_bert import ImageBertForSequenceClassificationwithAction

from transformers.pytorch_transformers import (
    BertTokenizer,
    BertConfig,
)


logger = logging.getLogger(__name__)

MODEL_CLASS = {
    "ImageBertForSequenceClassificationwithAction": (
        BertConfig,
        ImageBertForSequenceClassificationwithAction,
        BertTokenizer,
    )
}

special_tokens_dict = {
    "ques_token": "[QUES]",
    "ans_token": "[ANS]",
    "tar_token": "[TAR]",
}


def load_oscar_model(
    args, model_name, num_labels, add_new_extra_embeds=True, finetuned=False
):
    # Make sure only the first process in distributed training will download model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    config_class, model_class, tokenizer_class = MODEL_CLASS[model_name]
    config = config_class.from_pretrained(
        args.model_name_or_path, num_labels=num_labels, finetuning_task="NDH",
    )

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, do_lower_case=True,
    )

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

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

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

        model.resize_embeddings(embedding_size_dict)
    return model, tokenizer, config


def finetuned_model(args):
    pass
