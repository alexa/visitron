# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.pytorch_transformers import (AdamW, BertConfig,
                                               BertTokenizer,
                                               WarmupConstantSchedule,
                                               WarmupLinearSchedule,
                                               modeling_bert)

from agent import OscarAgent
from agent_lstm import LSTMAgent
from data_loader import VLNDataLoader, VLNDataloader_collate_fn, VLNDataset
from data_loader_lstm import (LSTMVLNDataLoader, LSTMVLNDataloader_collate_fn,
                              LSTMVLNDataset)
from eval import Evaluation
from params import args
from utils import Tokenizer, read_vocab, set_seed, setup_vocab
from utils_data import load_img_pickle_features, read_img_features, timeSince
from utils_model import MODEL_CLASS, load_oscar_model

logger = logging.getLogger(__name__)

TRAIN_VOCAB = "tasks/NDH/data/train_vocab.txt"
TRAINVAL_VOCAB = "tasks/NDH/data/trainval_vocab.txt"


def train(args, features, region_labels, candidate_features):

    num_labels = OscarAgent.n_outputs()

    model, tokenizer, config = load_oscar_model(
        args,
        "ImageBertForSequenceClassificationwithAction",
        num_labels,
        add_new_extra_embeds=False,
        finetuned=args.eval_only,
    )

    if args.pretrained_fixed:
        for params in model.parameters():
            params.requires_grad = False
        for params in model.classifier.parameters():
            params.requires_grad = True

    model.to(args.device)

    if not args.train_only:
        val_seen_dataset = VLNDataset(
            args=args,
            splits=["val_seen"],
            tokenizer=tokenizer,
            truncate_dialog=True,
            path_type=args.path_type,
            use_oscar_settings=True,
        )

        val_unseen_dataset = VLNDataset(
            args=args,
            splits=["val_unseen"],
            tokenizer=tokenizer,
            truncate_dialog=True,
            path_type=args.path_type,
            use_oscar_settings=True,
        )

        val_datasets = {
            "val_seen": val_seen_dataset,
            "val_unseen": val_unseen_dataset,
        }

    train_dataset = VLNDataset(
        args=args,
        splits=["train"],
        tokenizer=tokenizer,
        truncate_dialog=True,
        path_type=args.path_type,
        use_oscar_settings=True,
    )

    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(logdir=tensorboard_dir, flush_secs=30)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )

    train_data_loader = VLNDataLoader(
        dataset=train_dataset,
        feature_store=features,
        candidate_feature_store=candidate_features,
        region_labels=region_labels,
        tokenizer=tokenizer,
        batch_size=args.train_batch_size,
        collate_fn=VLNDataloader_collate_fn,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_data_loaders = {}
    if not args.eval_only:
        assert val_datasets is not None
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        for split, val_dataset in val_datasets.items():
            val_sampler = SequentialSampler(val_dataset)
            val_data_loader = VLNDataLoader(
                dataset=val_dataset,
                feature_store=features,
                candidate_feature_store=candidate_features,
                region_labels=region_labels,
                tokenizer=tokenizer,
                batch_size=args.eval_batch_size,
                collate_fn=VLNDataloader_collate_fn,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            evaluation = Evaluation([split], path_type=args.path_type)
            val_data_loaders[split] = (val_data_loader, evaluation)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    agent = OscarAgent(
        args=args,
        dataloader=train_data_loader,
        results_path="",
        model=model,
        episode_len=args.max_episode_len,
        use_oscar_settings=True,
    )

    logger.info("Training an Oscar agent with %s feedback" % args.feedback_method)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=args.num_iterations
        )
    else:
        raise ValueError("args.scheduler not found")

    data_log = defaultdict(list)
    start = time.time()

    n_iters = args.num_iterations
    log_every = args.logging_steps

    for idx in range(0, n_iters, log_every):
        interval = min(log_every, n_iters - idx)
        iter_no = idx + interval
        if args.local_rank in [-1, 0]:
            data_log["iteration"].append(iter_no)

        agent.train(optimizer, scheduler, interval, feedback=args.feedback_method)

        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)

        loss_str = ""

        if args.local_rank in [-1, 0]:
            # data_log["train loss"].append(train_loss_avg)
            data_log["train loss"].append(train_loss_avg)
            loss_str += "train loss: %.4f" % train_loss_avg
            # logger.info(f"Avg Loss: {train_loss_avg}")
            tb_writer.add_scalar("loss/train", train_loss_avg, global_step=iter_no)

        if (
            args.local_rank in [-1, 0]
            and args.evaluate_during_training
            and idx % args.eval_logging_steps == 0
        ):
            for env_name, (dataloader, evaluator) in val_data_loaders.items():
                agent.dataloader = dataloader
                agent.data_iter = iter(agent.dataloader)

                agent.results_path = os.path.join(
                    args.output_dir, "predictions", f"{env_name}-{iter_no}.json"
                )
                agent.test(
                    use_dropout=True, feedback=args.feedback_method, allow_cheat=True
                )
                val_losses = np.array(agent.losses)
                val_loss_avg = np.average(val_losses)
                tb_writer.add_scalar(
                    f"loss/{env_name}", val_loss_avg, global_step=iter_no
                )
                data_log["%s loss" % env_name].append(val_loss_avg)

                agent.test(use_dropout=False, feedback="argmax")
                agent.write_results()
                score_summary, _ = evaluator.score(agent.results_path)
                loss_str += ", %s loss: %.4f" % (env_name, val_loss_avg)
                for metric, val in score_summary.items():
                    data_log["%s %s" % (env_name, metric)].append(val)
                    if metric in [
                        "success_rate",
                        "oracle_success_rate",
                        "oracle_path_success_rate",
                        "dist_to_end_reduction",
                    ]:
                        loss_str += ", %s: %.3f" % (metric, val)
                        tb_writer.add_scalar(
                            f"{metric}/{env_name}", val, global_step=iter_no
                        )

            agent.dataloader = train_data_loader
            agent.data_iter = iter(agent.dataloader)

        if (
            args.local_rank in [-1, 0]
            and args.save_steps > 0
            and idx % args.save_steps == 0
        ):

            df = pd.DataFrame(data_log)
            df.set_index("iteration")
            df_path = os.path.join(args.output_dir, "results", f"{iter_no}-log.csv")
            df.to_csv(df_path)

            output_dir = os.path.join(
                args.output_dir, "checkpoints", f"checkpoint-{iter_no}"
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training

            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Saving model checkpoint {iter_no} to {output_dir}")

        logger.info(
            "%s (%d %d%%) %s"
            % (
                timeSince(start, float(iter_no) / n_iters),
                iter_no,
                float(iter_no) / n_iters * 100,
                loss_str,
            )
        )


def val(args, features, region_labels, candidate_features, list_iter_no):

    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(logdir=tensorboard_dir, flush_secs=30)

    num_labels = OscarAgent.n_outputs()

    root_folder = args.model_name_or_path

    for iter_no in list_iter_no:
        args.model_name_or_path = os.path.join(root_folder, f"checkpoint-{iter_no}")

        model, tokenizer, config = load_oscar_model(
            args,
            "ImageBertForSequenceClassificationwithAction",
            num_labels,
            add_new_extra_embeds=False,
            finetuned=args.eval_only,
        )
        model.to(args.device)

        val_seen_dataset = VLNDataset(
            args=args,
            splits=["val_seen"],
            tokenizer=tokenizer,
            truncate_dialog=True,
            path_type=args.path_type,
            use_oscar_settings=True,
        )

        val_unseen_dataset = VLNDataset(
            args=args,
            splits=["val_unseen"],
            tokenizer=tokenizer,
            truncate_dialog=True,
            path_type=args.path_type,
            use_oscar_settings=True,
        )

        val_datasets = {
            "val_seen": val_seen_dataset,
            "val_unseen": val_unseen_dataset,
        }

        val_data_loaders = {}
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        for split, val_dataset in val_datasets.items():
            val_sampler = SequentialSampler(val_dataset)
            val_data_loader = VLNDataLoader(
                dataset=val_dataset,
                feature_store=features,
                candidate_feature_store=candidate_features,
                region_labels=region_labels,
                tokenizer=tokenizer,
                batch_size=args.eval_batch_size,
                collate_fn=VLNDataloader_collate_fn,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            evaluation = Evaluation([split], path_type=args.path_type)
            val_data_loaders[split] = (val_data_loader, evaluation)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

        agent = OscarAgent(
            args=args,
            dataloader=val_data_loader,
            results_path="",
            model=model,
            episode_len=args.max_episode_len,
            use_oscar_settings=True,
        )

        logger.info(
            "Validating the Oscar agent with %s feedback for iteration %d"
            % (args.feedback_method, iter_no)
        )

        data_log = defaultdict(list)

        data_log["iteration"].append(iter_no)

        for env_name, (dataloader, evaluator) in val_data_loaders.items():
            start = time.time()
            agent.dataloader = dataloader
            agent.data_iter = iter(agent.dataloader)

            agent.results_path = os.path.join(
                args.output_dir, "predictions", f"{env_name}-{iter_no}.json"
            )
            agent.test(
                use_dropout=True, feedback=args.feedback_method, allow_cheat=True
            )
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            tb_writer.add_scalar(f"loss/{env_name}", val_loss_avg, global_step=iter_no)
            data_log["%s loss" % env_name].append(val_loss_avg)

            agent.test(use_dropout=False, feedback="argmax")
            agent.write_results()
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str = ", %s loss: %.4f" % (env_name, val_loss_avg)
            for metric, val in score_summary.items():
                data_log["%s %s" % (env_name, metric)].append(val)
                if metric in [
                    "length",
                    "nav_error",
                    "success_rate",
                    "oracle_success_rate",
                    "oracle_path_success_rate",
                    "spl",
                    "dist_to_end_reduction",
                ]:
                    loss_str += ", %s: %.3f" % (metric, val)
                    tb_writer.add_scalar(
                        f"{metric}/{env_name}", val, global_step=iter_no
                    )

            end = time.time()
            logger.info(
                "Time: %0.2f min Eval Iter: %d %s"
                % ((end - start) / 60, iter_no, loss_str,)
            )

        df = pd.DataFrame(data_log)
        df.set_index("iteration")
        df_path = os.path.join(args.output_dir, "results", f"{iter_no}-log.csv")
        df.to_csv(df_path)
    sys.exit()


def train_attn_lstm(args, features):
    if args.no_pretrained_model:
        setup_vocab(TRAIN_VOCAB, TRAINVAL_VOCAB)
        vocab = read_vocab(TRAIN_VOCAB)
        tokenizer = Tokenizer(vocab=vocab, encoding_length=args.max_seq_length)
    else:
        # num_labels = OscarAgent.n_outputs()
        model, tokenizer, config = load_oscar_model(
            args,
            "PreTrainImageBertForSequenceClassificationwithAction",
            add_new_extra_embeds=False,
            finetuned=args.eval_only,
        )

        if args.encoder == "lstm":
            word_embedding = model.bert.embeddings
        elif args.encoder == "oscar":
            bert_encoder = model.bert
        else:
            raise NotImplementedError

    train_dataset = LSTMVLNDataset(
        args=args,
        splits=["train"],
        tokenizer=tokenizer,
        truncate_dialog=True,
        path_type=args.path_type,
        add_ndh_data=True,
        add_r2r_data=args.add_r2r_data,
    )

    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(logdir=tensorboard_dir, flush_secs=30)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )

    train_data_loader = LSTMVLNDataLoader(
        dataset=train_dataset,
        splits=["train"],
        feature_store=features,
        tokenizer=tokenizer,
        batch_size=args.train_batch_size,
        collate_fn=LSTMVLNDataloader_collate_fn,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.no_pretrained_model:
        agent = LSTMAgent(
            args=args,
            tokenizer=tokenizer,
            vocab=vocab,
            dataloader=train_data_loader,
            results_path="",
            episode_len=args.max_episode_len,
        )
    else:
        if args.encoder == "lstm":
            agent = LSTMAgent(
                args=args,
                tokenizer=tokenizer,
                dataloader=train_data_loader,
                results_path="",
                word_embedding=word_embedding,
                episode_len=args.max_episode_len,
            )
        elif args.encoder == "oscar":
            agent = LSTMAgent(
                args=args,
                tokenizer=tokenizer,
                dataloader=train_data_loader,
                results_path="",
                bert=bert_encoder,
                episode_len=args.max_episode_len,
            )

    if args.n_gpu > 1:
        agent.encoder = torch.nn.DataParallel(agent.encoder)
        agent.decoder = torch.nn.DataParallel(agent.decoder)

    if args.local_rank != -1:
        agent.encoder = torch.nn.parallel.DistributedDataParallel(
            agent.encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        agent.decoder = torch.nn.parallel.DistributedDataParallel(
            agent.decoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    logger.info("Training an LSTM agent with %s feedback" % args.feedback_method)

    data_log = defaultdict(list)
    start = time.time()

    n_iters = args.num_iterations
    log_every = args.logging_steps

    for idx in range(0, n_iters, log_every):
        interval = min(log_every, n_iters - idx)
        iter_no = idx + interval
        if args.local_rank in [-1, 0]:
            data_log["iteration"].append(iter_no)

        agent.train(interval, feedback=args.feedback_method)

        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)

        loss_str = ""

        if args.local_rank in [-1, 0]:
            data_log["train loss"].append(train_loss_avg)
            loss_str += "train loss: %.4f" % train_loss_avg
            # logger.info(f"Avg Loss: {train_loss_avg}")
            tb_writer.add_scalar("loss/train", train_loss_avg, global_step=iter_no)

        if (
            args.local_rank in [-1, 0]
            and args.save_steps > 0
            and idx % args.save_steps == 0
        ):

            df = pd.DataFrame(data_log)
            df.set_index("iteration")
            df_path = os.path.join(args.output_dir, "results", f"{iter_no}-log.csv")
            df.to_csv(df_path)

            output_dir = os.path.join(
                args.output_dir, "checkpoints", f"checkpoint-{iter_no}"
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            agent.save(
                os.path.join(output_dir, "encoder"), os.path.join(output_dir, "decoder")
            )
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            if not args.no_pretrained_model:
                tokenizer.save_pretrained(output_dir)
            logger.info(f"Saving model checkpoint {iter_no} to {output_dir}")

        logger.info(
            "%s (%d %d%%) %s"
            % (
                timeSince(start, float(iter_no) / n_iters),
                iter_no,
                float(iter_no) / n_iters * 100,
                loss_str,
            )
        )


def val_attn_lstm(args, features, list_iter_no):

    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(logdir=tensorboard_dir, flush_secs=30)

    if not args.no_pretrained_model:
        num_labels = OscarAgent.n_outputs()

    root_folder = args.model_name_or_path

    for iter_no in list_iter_no:

        encoder_path = os.path.join(root_folder, f"checkpoint-{iter_no}", "encoder")
        decoder_path = os.path.join(root_folder, f"checkpoint-{iter_no}", "decoder")
        if not args.no_pretrained_model:
            tokenizer_path = os.path.join(root_folder, f"checkpoint-{iter_no}/")

            tmp_root_folder = (
                "oscar_ndh_results/viewpoint_selection_v1/checkpoints/checkpoint-22510"
            )
            config_path = os.path.join(tmp_root_folder, "config.json")

            config = BertConfig.from_pretrained(config_path)

            config.action_space = 36

            if args.encoder == "lstm":
                word_embedding = modeling_bert.BertEmbeddings(config)
            elif args.encoder == "oscar":
                model_class = MODEL_CLASS[
                    "PreTrainImageBertForSequenceClassificationwithAction"
                ][1]
                model = model_class(config)
                bert_encoder = model.bert
            else:
                raise NotImplementedError

            tokenizer = BertTokenizer.from_pretrained(
                tokenizer_path, do_lower_case=True,
            )
        else:
            setup_vocab(TRAIN_VOCAB, TRAINVAL_VOCAB)
            vocab = read_vocab(TRAIN_VOCAB)
            tokenizer = Tokenizer(vocab=vocab, encoding_length=args.max_seq_length)

        val_seen_dataset = LSTMVLNDataset(
            args=args,
            splits=["val_seen"],
            tokenizer=tokenizer,
            truncate_dialog=True,
            path_type=args.path_type,
        )

        val_unseen_dataset = LSTMVLNDataset(
            args=args,
            splits=["val_unseen"],
            tokenizer=tokenizer,
            truncate_dialog=True,
            path_type=args.path_type,
        )

        val_datasets = {
            "val_seen": val_seen_dataset,
            "val_unseen": val_unseen_dataset,
        }

        val_data_loaders = {}
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        for split, val_dataset in val_datasets.items():
            val_sampler = SequentialSampler(val_dataset)
            val_data_loader = LSTMVLNDataLoader(
                dataset=val_dataset,
                splits=[split],
                feature_store=features,
                tokenizer=tokenizer,
                batch_size=args.eval_batch_size,
                collate_fn=LSTMVLNDataloader_collate_fn,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            evaluation = Evaluation([split], path_type=args.path_type)
            val_data_loaders[split] = (val_data_loader, evaluation)

        if args.no_pretrained_model:
            agent = LSTMAgent(
                args=args,
                tokenizer=tokenizer,
                vocab=vocab,
                dataloader=val_data_loader,
                results_path="",
                episode_len=args.max_episode_len,
            )
        else:
            if args.encoder == "lstm":
                agent = LSTMAgent(
                    args=args,
                    tokenizer=tokenizer,
                    dataloader=val_data_loader,
                    results_path="",
                    word_embedding=word_embedding,
                    episode_len=args.max_episode_len,
                )
            elif args.encoder == "oscar":
                agent = LSTMAgent(
                    args=args,
                    tokenizer=tokenizer,
                    dataloader=val_data_loader,
                    results_path="",
                    bert=bert_encoder,
                    episode_len=args.max_episode_len,
                )

        agent.load(encoder_path, decoder_path)

        if args.n_gpu > 1:
            agent.encoder = torch.nn.DataParallel(agent.encoder)
            agent.decoder = torch.nn.DataParallel(agent.decoder)

        if args.local_rank != -1:
            agent.encoder = torch.nn.parallel.DistributedDataParallel(
                agent.encoder,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )
            agent.decoder = torch.nn.parallel.DistributedDataParallel(
                agent.decoder,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

        logger.info(
            "Validating the LSTM agent with %s feedback for iteration %d"
            % (args.feedback_method, iter_no)
        )

        data_log = defaultdict(list)

        data_log["iteration"].append(iter_no)

        for env_name, (dataloader, evaluator) in val_data_loaders.items():
            start = time.time()
            agent.dataloader = dataloader
            agent.data_iter = iter(agent.dataloader)

            agent.results_path = os.path.join(
                args.output_dir, "predictions", f"{env_name}-{iter_no}.json"
            )

            agent.test(use_dropout=True, feedback=args.feedback_method, allow_cheat=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            tb_writer.add_scalar(f"loss/{env_name}", val_loss_avg, global_step=iter_no)
            data_log["%s loss" % env_name].append(val_loss_avg)

            agent.test(use_dropout=False, feedback="argmax")
            agent.write_results()

            score_summary, _ = evaluator.score(agent.results_path)
            loss_str = ", %s loss: %.4f" % (env_name, val_loss_avg)
            for metric, val in score_summary.items():
                data_log["%s %s" % (env_name, metric)].append(val)
                if metric in [
                    "length",
                    "hops",
                    "nav_error",
                    "success_rate",
                    "oracle_success_rate",
                    "oracle_path_success_rate",
                    "spl",
                    "dist_to_end_reduction",
                ]:
                    loss_str += ", %s: %.3f" % (metric, val)
                    tb_writer.add_scalar(
                        f"{metric}/{env_name}", val, global_step=iter_no
                    )

            end = time.time()
            logger.info(
                "Time: %0.2f min Eval Iter: %d %s"
                % ((end - start) / 60, iter_no, loss_str,)
            )

        df = pd.DataFrame(data_log)
        df.set_index("iteration")
        df_path = os.path.join(args.output_dir, "results", f"{iter_no}-log.csv")
        df.to_csv(df_path)
    sys.exit()


def test_submission(args, features, list_iter_no):

    if not args.no_pretrained_model:
        num_labels = OscarAgent.n_outputs()

    root_folder = args.model_name_or_path

    for iter_no in list_iter_no:

        encoder_path = os.path.join(root_folder, f"checkpoint-{iter_no}", "encoder")
        decoder_path = os.path.join(root_folder, f"checkpoint-{iter_no}", "decoder")
        if not args.no_pretrained_model:
            tokenizer_path = os.path.join(root_folder, f"checkpoint-{iter_no}/")

            tmp_root_folder = (
                "oscar_ndh_results/viewpoint_selection_v1/checkpoints/checkpoint-22510"
            )
            config_path = os.path.join(tmp_root_folder, "config.json")

            config = BertConfig.from_pretrained(config_path)

            config.action_space = 36

            if args.encoder == "lstm":
                word_embedding = modeling_bert.BertEmbeddings(config)
            elif args.encoder == "oscar":
                model_class = MODEL_CLASS[
                    "PreTrainImageBertForSequenceClassificationwithAction"
                ][1]
                model = model_class(config)
                bert_encoder = model.bert
            else:
                raise NotImplementedError

            tokenizer = BertTokenizer.from_pretrained(
                tokenizer_path, do_lower_case=True,
            )
        else:
            setup_vocab(TRAIN_VOCAB, TRAINVAL_VOCAB)
            vocab = read_vocab(TRAIN_VOCAB)
            tokenizer = Tokenizer(vocab=vocab, encoding_length=args.max_seq_length)

        test_dataset = LSTMVLNDataset(
            args=args,
            splits=["test"],
            tokenizer=tokenizer,
            truncate_dialog=True,
            path_type=args.path_type,
        )

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        test_sampler = SequentialSampler(test_dataset)
        test_data_loader = LSTMVLNDataLoader(
            dataset=test_dataset,
            splits=["test"],
            feature_store=features,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            collate_fn=LSTMVLNDataloader_collate_fn,
            sampler=test_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        evaluation = Evaluation(["test"], path_type=args.path_type)
        # val_data_loaders[split] = (val_data_loader, evaluation)

        if args.no_pretrained_model:
            agent = LSTMAgent(
                args=args,
                tokenizer=tokenizer,
                vocab=vocab,
                dataloader=test_data_loader,
                results_path="",
                episode_len=args.max_episode_len,
            )
        else:
            if args.encoder == "lstm":
                agent = LSTMAgent(
                    args=args,
                    tokenizer=tokenizer,
                    dataloader=test_data_loader,
                    results_path="",
                    word_embedding=word_embedding,
                    episode_len=args.max_episode_len,
                )
            elif args.encoder == "oscar":
                agent = LSTMAgent(
                    args=args,
                    tokenizer=tokenizer,
                    dataloader=test_data_loader,
                    results_path="",
                    bert=bert_encoder,
                    episode_len=args.max_episode_len,
                )

        agent.load(encoder_path, decoder_path)

        if args.n_gpu > 1:
            agent.encoder = torch.nn.DataParallel(agent.encoder)
            agent.decoder = torch.nn.DataParallel(agent.decoder)

        if args.local_rank != -1:
            agent.encoder = torch.nn.parallel.DistributedDataParallel(
                agent.encoder,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )
            agent.decoder = torch.nn.parallel.DistributedDataParallel(
                agent.decoder,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

        logger.info(
            "Generating test spplit predictions for the LSTM agent iteration %d"
            % (iter_no)
        )

        data_log = defaultdict(list)

        data_log["iteration"].append(iter_no)

        start = time.time()
        agent.dataloader = test_data_loader
        agent.data_iter = iter(agent.dataloader)

        agent.results_path = os.path.join(
            args.output_dir, "predictions", f"test-{iter_no}.json"
        )

        agent.test(use_dropout=False, feedback="argmax")
        agent.write_results()

        # score_summary, _ = evaluation.score(agent.results_path)
        # logger.info(score_summary)

        end = time.time()
        logger.info(
            "Time: %0.2f min Eval Iter: %d %s"
            % ((end - start) / 60, iter_no, loss_str,)
        )

        df = pd.DataFrame(data_log)
        df.set_index("iteration")
        df_path = os.path.join(args.output_dir, "results", f"{iter_no}-log.csv")
        df.to_csv(df_path)
    sys.exit()


def main():

    if (
        args.local_rank in [-1, 0]
        and os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.eval_only
    ):
        raise IOError(
            "%s \nOutput Directory not empty and train setting is on. Exiting to prevent overwriting..."
            % (args.output_dir)
        )

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"))
        os.makedirs(os.path.join(args.output_dir, "predictions"))
        os.makedirs(os.path.join(args.output_dir, "results"))
        os.makedirs(os.path.join(args.output_dir, "tensorboard"))

    handlers = [logging.StreamHandler()]
    if args.local_rank in [-1, 0]:
        handlers += [
            logging.FileHandler(filename=os.path.join(args.output_dir, "log"), mode="a")
        ]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        handlers=handlers,
    )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    if args.path_type == "planner_path":
        args.max_episode_len = 10
    else:
        args.max_episode_len = 40

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        "False",
    )

    # Set seed
    set_seed(args.seed, args.n_gpu)

    logger.info("Training/evaluation parameters %s", args)

    if args.agent == "oscar":
        features, region_labels, candidate_features = load_img_pickle_features(
            args.img_feat_dir, args.img_feature_file, args.candidate_feature_file
        )

        if args.eval_only:
            assert (
                len(args.eval_iters) != 0 and args.eval_iters != -1
            ), "incorrect eval_iters provided!"
            val(args, features, region_labels, candidate_features, args.eval_iters)
        else:
            train(args, features, region_labels, candidate_features)

    elif args.agent == "att-lstm":
        features = read_img_features(
            os.path.join(args.img_feat_dir, args.img_feature_file)
        )
        if args.test_only:
            assert (
                len(args.eval_iters) != 0 and args.eval_iters != -1
            ), "incorrect eval_iters provided!"
            test_submission(args, features, args.eval_iters)
        if args.eval_only:
            assert (
                len(args.eval_iters) != 0 and args.eval_iters != -1
            ), "incorrect eval_iters provided!"
            val_attn_lstm(args, features, args.eval_iters)
        else:
            train_attn_lstm(args, features)

    sys.exit()


if __name__ == "__main__":
    main()
                                                                                                        