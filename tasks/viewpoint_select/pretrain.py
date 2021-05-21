# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import logging
import os
import sys
import time
from collections import defaultdict

import pandas as pd
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data_loader_pretrain import PretrainDataset
from params import args
from utils import set_seed
from utils_data import FeaturesReader, timeSince

sys.path.insert(0, "/root/mount/Matterport3DSimulator/")

from get_oscar_model import load_oscar_model
from oscar.transformers_src.pytorch_transformers import (
    AdamW,
    WarmupConstantSchedule,
    WarmupLinearSchedule,
)

logger = logging.getLogger(__name__)


def train(args, features_reader):

    model, tokenizer, config = load_oscar_model(
        args,
        "PreTrainOscar",
        add_new_extra_embeds=(not args.oscar_setting),
        finetuned=args.eval_only,
    )

    if args.pretrained_fixed:
        for params in model.parameters():
            params.requires_grad = False
        for params in model.classifier.parameters():
            params.requires_grad = True

    model.to(args.device)

    version = "v3"  # tar front
    if args.tar_back:
        version = "v4"  # tar back
    if args.oscar_setting is False:
        version = "v5"  # adding new embeds
    if args.masked_token_prediction:
        version = "v2"

    train_dataset = PretrainDataset(
        args=args,
        splits=["train"],
        features_reader=features_reader,
        tokenizer=tokenizer,
        truncate_dialog=True,
        add_ndh_data=args.add_ndh_data,
        add_r2r_data=args.add_r2r_data,
        add_r4r_data=args.add_r4r_data,
        add_rxr_data=args.add_rxr_data,
        version=version,
    )

    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    if args.local_rank in [-2, -1, 0]:
        tb_writer = SummaryWriter(logdir=tensorboard_dir, flush_secs=30)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank in [-2, -1]
        else DistributedSampler(train_dataset)
    )

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank not in [-2, -1]:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    logger.info(
        "Pretraining the Oscar model for Masked LM and 1-in-36 Viewpoint Prediction"
    )

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
    global_iter = -1

    iters_per_epoch = len(train_data_loader)
    total_iters = args.num_epochs * iters_per_epoch

    for epoch_no in range(args.num_epochs):

        for step, batch in enumerate(train_data_loader):

            global_iter += 1
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch = {key: item.to(args.device) for key, item in batch.items()}

            (
                loss,
                mask_loss,
                next_loss,
                token_loss,
                words_accuracy,
                action_accuracy,
                token_accuracy,
            ) = model(**batch)

            if args.local_rank not in [-2, -1]:
                loss /= dist.get_world_size()
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)

                mask_loss /= dist.get_world_size()
                dist.all_reduce(mask_loss, op=dist.ReduceOp.SUM)

                next_loss /= dist.get_world_size()
                dist.all_reduce(next_loss, op=dist.ReduceOp.SUM)

                token_loss /= dist.get_world_size()
                dist.all_reduce(token_loss, op=dist.ReduceOp.SUM)

                words_accuracy /= dist.get_world_size()
                dist.all_reduce(words_accuracy, op=dist.ReduceOp.SUM)

                action_accuracy /= dist.get_world_size()
                dist.all_reduce(action_accuracy, op=dist.ReduceOp.SUM)

                token_accuracy /= dist.get_world_size()
                dist.all_reduce(token_accuracy, op=dist.ReduceOp.SUM)

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss = loss.cpu().detach().item()
            mask_loss = mask_loss.cpu().detach().item()
            next_loss = next_loss.cpu().detach().item()
            token_loss = token_loss.cpu().detach().item()
            words_accuracy = words_accuracy.cpu().detach().item()
            action_accuracy = action_accuracy.cpu().detach().item()
            token_accuracy = token_accuracy.cpu().detach().item()

            if args.local_rank in [-2, -1, 0] and (
                global_iter % args.logging_steps == 0 or step == iters_per_epoch - 1
            ):
                data_log["epoch_no"].append(epoch_no)
                data_log["iteration"].append(step)
                data_log["global_iter"].append(global_iter)
                data_log["train_loss"].append(loss)
                data_log["mask_loss"].append(mask_loss)
                data_log["token_loss"].append(token_loss)
                data_log["next_action_loss"].append(next_loss)
                data_log["words_accuracy"].append(words_accuracy)
                data_log["action_accuracy"].append(action_accuracy)
                data_log["token_accuracy"].append(token_accuracy)

                tb_writer.add_scalar("loss/train_all", loss, global_step=global_iter)
                tb_writer.add_scalar(
                    "loss/train_mask", mask_loss, global_step=global_iter
                )
                tb_writer.add_scalar(
                    "loss/train_next_action", next_loss, global_step=global_iter
                )
                tb_writer.add_scalar(
                    "loss/train_token", token_loss, global_step=global_iter
                )
                tb_writer.add_scalar(
                    "accuracy/train_word", words_accuracy, global_step=global_iter
                )
                tb_writer.add_scalar(
                    "accuracy/train_action",
                    action_accuracy,
                    global_step=global_iter,
                )
                tb_writer.add_scalar(
                    "accuracy/train_token",
                    token_accuracy,
                    global_step=global_iter,
                )
                log_str = (
                    f"Global Iter: {global_iter} Epoch: {epoch_no}/{args.num_epochs} Iter: {step}/{iters_per_epoch}"
                    + f"\t Train Loss: {loss:.04} Mask Loss: {mask_loss:.04} Next Action Loss: {next_loss:.04} Token Prediction Loss: {token_loss:.04} Word Acc: {words_accuracy:.02} Action Acc: {action_accuracy:.02} Token Acc: {token_accuracy:.02}"
                    + f"\t {timeSince(start, float(global_iter+1) / total_iters)}"
                )
                logger.info(log_str)

            if (args.local_rank in [-2, -1, 0] and args.save_steps > 0) and (
                global_iter % args.save_steps == 0 or step == iters_per_epoch - 1
            ):
                df = pd.DataFrame(data_log)
                df.set_index("global_iter")
                df_path = os.path.join(
                    args.output_dir, "results", f"{global_iter}-log.csv"
                )
                df.to_csv(df_path)

                output_dir = os.path.join(
                    args.output_dir, "checkpoints", f"checkpoint-{global_iter}"
                )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training

                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                tokenizer.save_pretrained(output_dir)
                logger.info(f"Saving model checkpoint {global_iter} to {output_dir}")


def val(args, features_reader, list_iter_no):

    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    if args.local_rank in [-2, -1, 0]:
        tb_writer = SummaryWriter(logdir=tensorboard_dir, flush_secs=30)

    root_folder = args.model_name_or_path

    for iter_no in list_iter_no:
        args.model_name_or_path = os.path.join(root_folder, f"checkpoint-{iter_no}")

        model, tokenizer, config = load_oscar_model(
            args,
            "PreTrainOscar",
            add_new_extra_embeds=(not args.oscar_setting),
            finetuned=args.eval_only,
        )
        model.to(args.device)
        model.eval()

        version = "v3"  # tar front
        if args.tar_back:
            version = "v4"  # tar back
        if args.oscar_setting is False:
            version = "v5"  # adding new embeds
        if args.masked_token_prediction:
            version = "v2"

        ndh_val_seen_dataset = PretrainDataset(
            args=args,
            splits=["val_seen"],
            features_reader=features_reader,
            tokenizer=tokenizer,
            truncate_dialog=True,
            add_ndh_data=True,
            add_r2r_data=False,
            add_r4r_data=False,
            add_rxr_data=False,
            version=version,
        )

        ndh_val_unseen_dataset = PretrainDataset(
            args=args,
            splits=["val_unseen"],
            features_reader=features_reader,
            tokenizer=tokenizer,
            truncate_dialog=True,
            add_ndh_data=True,
            add_r2r_data=False,
            add_r4r_data=False,
            add_rxr_data=False,
            version=version,
        )

        val_datasets = {
            "ndh_val_seen": ndh_val_seen_dataset,
            "ndh_val_unseen": ndh_val_unseen_dataset,
        }

        if args.add_r2r_data:
            r2r_val_seen_dataset = PretrainDataset(
                args=args,
                splits=["val_seen"],
                features_reader=features_reader,
                tokenizer=tokenizer,
                truncate_dialog=True,
                add_ndh_data=False,
                add_r2r_data=True,
                add_r4r_data=False,
                add_rxr_data=False,
                version=version,
            )

            r2r_val_unseen_dataset = PretrainDataset(
                args=args,
                splits=["val_unseen"],
                features_reader=features_reader,
                tokenizer=tokenizer,
                truncate_dialog=True,
                add_ndh_data=False,
                add_r2r_data=True,
                add_r4r_data=False,
                add_rxr_data=False,
                version=version,
            )

            val_datasets["r2r_val_seen"] = r2r_val_seen_dataset
            val_datasets["r2r_val_unseen"] = r2r_val_unseen_dataset

        if args.add_r4r_data:
            r4r_val_seen_dataset = PretrainDataset(
                args=args,
                splits=["val_seen"],
                features_reader=features_reader,
                tokenizer=tokenizer,
                truncate_dialog=True,
                add_ndh_data=False,
                add_r2r_data=False,
                add_r4r_data=True,
                add_rxr_data=False,
                version=version,
            )

            r4r_val_unseen_dataset = PretrainDataset(
                args=args,
                splits=["val_unseen"],
                features_reader=features_reader,
                tokenizer=tokenizer,
                truncate_dialog=True,
                add_ndh_data=False,
                add_r2r_data=False,
                add_r4r_data=True,
                add_rxr_data=False,
                version=version,
            )

            val_datasets["r4r_val_seen"] = r4r_val_seen_dataset
            val_datasets["r4r_val_unseen"] = r4r_val_unseen_dataset

        if args.add_rxr_data:
            rxr_val_seen_dataset = PretrainDataset(
                args=args,
                splits=["val_seen"],
                features_reader=features_reader,
                tokenizer=tokenizer,
                truncate_dialog=True,
                add_ndh_data=False,
                add_r2r_data=False,
                add_r4r_data=False,
                add_rxr_data=True,
                version=version,
            )

            rxr_val_unseen_dataset = PretrainDataset(
                args=args,
                splits=["val_unseen"],
                features_reader=features_reader,
                tokenizer=tokenizer,
                truncate_dialog=True,
                add_ndh_data=False,
                add_r2r_data=False,
                add_r4r_data=False,
                add_rxr_data=True,
                version=version,
            )

            val_datasets["rxr_val_seen"] = rxr_val_seen_dataset
            val_datasets["rxr_val_unseen"] = rxr_val_unseen_dataset

        val_data_loaders = {}
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        for split, val_dataset in val_datasets.items():
            val_sampler = SequentialSampler(val_dataset)
            val_data_loader = DataLoader(
                dataset=val_dataset,
                batch_size=args.eval_batch_size,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            val_data_loaders[split] = val_data_loader

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if args.local_rank not in [-2, -1]:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

        logger.info(
            "Validating the Oscar model for global iter %d for Masked LM and 1-in-36 Viewpoint Prediction"
            % (iter_no)
        )

        data_log = defaultdict(list)

        data_log["global_iter"].append(iter_no)

        for env_name, dataloader in val_data_loaders.items():
            start = time.time()

            total_loss = 0
            total_mask_loss = 0
            total_next_loss = 0
            total_token_loss = 0
            total_words_accuracy = 0
            total_action_accuracy = 0
            total_token_accuracy = 0

            total_count = 0

            for step, batch in tqdm(
                enumerate(dataloader), desc=f"Evaluating {env_name}"
            ):
                batch = {key: item.to(args.device) for key, item in batch.items()}
                (
                    loss,
                    mask_loss,
                    next_loss,
                    token_loss,
                    words_accuracy,
                    action_accuracy,
                    token_accuracy,
                ) = model(**batch)

                if args.local_rank not in [-2, -1]:
                    loss /= dist.get_world_size()
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)

                    mask_loss /= dist.get_world_size()
                    dist.all_reduce(mask_loss, op=dist.ReduceOp.SUM)

                    next_loss /= dist.get_world_size()
                    dist.all_reduce(next_loss, op=dist.ReduceOp.SUM)

                    token_loss /= dist.get_world_size()
                    dist.all_reduce(token_loss, op=dist.ReduceOp.SUM)

                    words_accuracy /= dist.get_world_size()
                    dist.all_reduce(words_accuracy, op=dist.ReduceOp.SUM)

                    action_accuracy /= dist.get_world_size()
                    dist.all_reduce(action_accuracy, op=dist.ReduceOp.SUM)

                    token_accuracy /= dist.get_world_size()
                    dist.all_reduce(token_accuracy, op=dist.ReduceOp.SUM)

                loss = loss.cpu().detach().item()
                mask_loss = mask_loss.cpu().detach().item()
                next_loss = next_loss.cpu().detach().item()
                token_loss = token_loss.cpu().detach().item()
                words_accuracy = words_accuracy.cpu().detach().item()
                action_accuracy = action_accuracy.cpu().detach().item()
                token_accuracy = token_accuracy.cpu().detach().item()

                total_loss += loss
                total_mask_loss += mask_loss
                total_next_loss += next_loss
                total_token_loss += token_loss
                total_words_accuracy += words_accuracy
                total_action_accuracy += action_accuracy
                total_token_accuracy += token_accuracy

                total_count += 1

            total_loss /= total_count
            total_mask_loss /= total_count
            total_next_loss /= total_count
            total_token_loss /= total_count
            total_words_accuracy /= total_count
            total_action_accuracy /= total_count
            total_token_accuracy /= total_count

            if args.local_rank in [-2, -1, 0]:
                data_log[f"{env_name}_loss"].append(total_loss)
                data_log[f"{env_name}_mask_loss"].append(total_mask_loss)
                data_log[f"{env_name}_next_action_loss"].append(total_next_loss)
                data_log[f"{env_name}_token_loss"].append(total_token_loss)
                data_log[f"{env_name}_words_accuracy"].append(total_words_accuracy)
                data_log[f"{env_name}_action_accuracy"].append(total_action_accuracy)
                data_log[f"{env_name}_token_accuracy"].append(total_token_accuracy)

                tb_writer.add_scalar(
                    f"loss/{env_name}_all", total_loss, global_step=iter_no
                )
                tb_writer.add_scalar(
                    f"loss/{env_name}_mask", total_mask_loss, global_step=iter_no
                )
                tb_writer.add_scalar(
                    f"loss/{env_name}_next_action", total_next_loss, global_step=iter_no
                )
                tb_writer.add_scalar(
                    f"loss/{env_name}_token", total_token_loss, global_step=iter_no
                )
                tb_writer.add_scalar(
                    f"accuracy/{env_name}_word",
                    total_words_accuracy,
                    global_step=iter_no,
                )
                tb_writer.add_scalar(
                    f"accuracy/{env_name}_action",
                    total_action_accuracy,
                    global_step=iter_no,
                )
                tb_writer.add_scalar(
                    f"accuracy/{env_name}_token",
                    total_token_accuracy,
                    global_step=iter_no,
                )
                end = time.time()
                log_str = (
                    f"Global Iter: {iter_no}"
                    + f"\t {env_name} Loss: {total_loss:.04} Mask Loss: {total_mask_loss:.04} Next Action Loss: {total_next_loss:.04} Token Prediction Loss: {token_loss:.04} Word Acc: {total_words_accuracy:.02} Action Acc: {total_action_accuracy:.02} Token Acc: {token_accuracy:.02}"
                    + f"\t Time: {(end-start)/60} mins"
                )
                logger.info(log_str)

        df = pd.DataFrame(data_log)
        df.set_index("global_iter")
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

    # Setup CPU, CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cpu")
        args.local_rank = -2
        args.n_gpu = -1
        if torch.cuda.is_available():
            device = torch.device("cuda")
            args.local_rank = -1
            args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logger.info(args.slurm_info)

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

    logger.info("Pretraining parameters %s", args)

    img_feature_path = os.path.join(args.img_feat_dir, args.img_feature_file)
    features_reader = FeaturesReader(
        path=img_feature_path,
        use_lmdb=(not args.eval_only),
        in_memory=False,
    )

    if args.eval_only:
        assert (
            len(args.eval_iters) != 0 and args.eval_iters != -1
        ), "incorrect eval_iters provided!"
        val(args, features_reader, args.eval_iters)
    else:
        train(args, features_reader)

    sys.exit()


if __name__ == "__main__":
    main()
