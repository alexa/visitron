import copy
import logging
import os
import random
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch import optim
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# from torch.autograd import Variable
from tqdm import tqdm

# from model import AttnDecoderLSTM, Critic, EncoderLSTM
import model
from gameplay.agent import Agent
from gameplay.data_loader import (R2RDataLoader, R2RDataLoader_collate_fn,
                                  R2RDataset)
from gameplay.eval import Evaluation
from gameplay.speaker import Speaker
from get_oscar_model import MODEL_CLASS, load_oscar_model, special_tokens_dict
from oscar.transformers_src.pytorch_transformers import (
    AdamW, BertConfig, BertTokenizer, WarmupConstantSchedule,
    WarmupLinearSchedule, modeling_bert)
from params import args
from utils import (Tokenizer, build_vocab, load, padding_idx, read_vocab, save,
                   set_seed, setup_vocab, write_vocab)
from utils_data import load_detector_classes, read_tsv_img_features, timeSince

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

logger.info(args)  # Print finalized args


def train_speaker(args, features):
    vocab = read_vocab(args.train_vocab)
    tok = Tokenizer(vocab=vocab, encoding_length=args.max_seq_length)

    train_dataset = R2RDataset(
        splits=["train"],
        tokenizer=tok,
        path_type=args.path_type,
        history=args.history,
    )

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank in [-2, -1]
        else DistributedSampler(train_dataset)
    )

    train_data_loader = R2RDataLoader(
        dataset=train_dataset,
        feature_store=features,
        batch_size=args.train_batch_size,
        collate_fn=R2RDataLoader_collate_fn,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_splits = ["val_seen", "val_unseen"]

    val_envs = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if args.local_rank in [-2, -1, 0]:
        # Create validation environments
        for split in val_splits:
            val_dataset = R2RDataset(
                splits=[split],
                tokenizer=tok,
                path_type=args.path_type,
                history=args.history,
            )
            evaluator = Evaluation(
                [split],
                path_type=args.path_type,
                results_dir=args.agent_results_dir,
                steps_to_next_q=args.steps_to_next_q,
            )

            val_sampler = SequentialSampler(val_dataset)
            val_data_loader = R2RDataLoader(
                dataset=val_dataset,
                feature_store=features,
                batch_size=args.eval_batch_size,
                collate_fn=R2RDataLoader_collate_fn,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            val_envs[split] = (val_data_loader, evaluator)

    # Build models and train
    # enc_hidden_size = args.HIDDEN_SIZE // 2 if args.BIDIRECTIONAL else args.HIDDEN_SIZE
    # encoder = EncoderLSTM(
    #     tok.vocab_size(),
    #     args.WORD_EMBEDDING_SIZE,
    #     enc_hidden_size,
    #     padding_idx,
    #     args.DROPOUT_RATIO,
    #     bidirectional=args.BIDIRECTIONAL,
    # ).to(args.device)
    # decoder = AttnDecoderLSTM(
    #     Seq2SeqAgent.n_inputs(),
    #     Seq2SeqAgent.n_outputs(),
    #     args.ACTION_EMBEDDING_SIZE,
    #     args.HIDDEN_SIZE,
    #     args.DROPOUT_RATIO,
    # ).to(args.device)

    # if args.n_gpu > 1:
    #     encoder = torch.nn.DataParallel(encoder)
    #     decoder = torch.nn.DataParallel(decoder)

    # if args.local_rank not in [-2, -1]:
    #     encoder = torch.nn.parallel.DistributedDataParallel(
    #         encoder,
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank,
    #         find_unused_parameters=True,
    #     )
    #     decoder = torch.nn.parallel.DistributedDataParallel(
    #         decoder,
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank,
    #         find_unused_parameters=True,
    #     )
    # encoder_optimizer = optim.Adam(
    #     encoder.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY
    # )
    # decoder_optimizer = optim.Adam(
    #     decoder.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY
    # )

    agent = Agent(
        args=args,
        dataloader=train_data_loader,
        results_path="",
        encoder=None,
        encoder_optimizer=None,
        decoder=None,
        decoder_optimizer=None,
        train_episode_len=None,
        eval_episode_len=None,
        turn_based=None,
        current_q_a_only=None,
        critic=None,
        critic_optimizer=None,
        use_rl=None,
        agent_rl=None,
        random_start=None,
        J=None,
        steps_to_next_q=None,
        train_branching=None,
        action_probs_branching=None,
    )

    speaker = Speaker(
        args=args,
        dataloader=train_data_loader,
        agent=agent,
        feedback=args.speaker_feedback_method,
        maxDecode=args.max_seq_length,
        speaker_rl=args.speaker_rl,
        tokenizer=tok,
    )

    logger.info("Training a speaker with %s feedback" % (args.speaker_feedback_method))

    start_iter, best_iter = 0, 0
    best_bleu = defaultdict(lambda: 0)

    if args.local_rank in [-2, -1, 0]:
        # Set up to save best model

        # best_loss = defaultdict(lambda: 1232)

        data_log = defaultdict(list)
        best_model = {
            "iter": 0,
            "encoder": copy.deepcopy(speaker.encoder),
            "encoder_optm": copy.deepcopy(speaker.encoder_optimizer),
            "decoder": copy.deepcopy(speaker.decoder),
            "decoder_optm": copy.deepcopy(speaker.decoder_optimizer),
        }

    # Run train-eval loop
    for idx in tqdm(
        range(start_iter, args.num_iterations, args.logging_steps),
        desc="Train-Eval Loop",
    ):
        # for idx in range(start_iter, args.num_iterations, args.logging_steps):

        interval = min(args.logging_steps, args.num_iterations - idx)
        iteration = idx + interval
        if args.local_rank in [-2, -1, 0]:
            data_log["iteration"].append(iteration)
            bleu_unseen = 0

        # Train for log_every interval
        speaker.dataloader = train_data_loader
        print("start train")
        speaker.train(1, train_nav=True)  # Train nav
        speaker.train(interval)  # Train ora
        print("end train")

        logger.info("Iter: %d" % idx)

        if args.local_rank in [-2, -1, 0]:
            # Evaluation
            for env_name, (dataloader, evaluator) in val_envs.items():
                logger.info("............ Evaluating %s ............." % env_name)

                speaker.dataloader = dataloader
                for instr_type in ["nav", "ora"]:
                    for_nav = instr_type == "nav"
                    logger.info("............ %s ............." % instr_type)
                    path2inst, loss, word_accu, sent_accu = speaker.valid(
                        for_nav=for_nav
                    )
                    path_id = next(iter(path2inst.keys()))
                    logger.info("\n\n")
                    logger.info(f"Inference: {tok.decode_sentence(path2inst[path_id])}")
                    logger.info(
                        f"GT: {evaluator.gt[path_id][instr_type + '_instructions']}"
                    )
                    bleu_score, precisions = evaluator.bleu_score(
                        path2inst, tok=tok, for_nav=for_nav
                    )

                    if not for_nav:
                        data_log["%s bleu" % (env_name)].append(bleu_score)
                        data_log["%s loss" % (env_name)].append(loss)

                        # Save the model according to the bleu score
                        if bleu_score > best_bleu[env_name]:
                            best_bleu[env_name] = bleu_score
                            if env_name == "val_unseen":
                                logger.info(
                                    "Save the model with %s BEST env bleu %0.4f"
                                    % (env_name, bleu_score)
                                )
                                bleu_unseen = bleu_score
                                best_iter = iteration
                                best_model = {
                                    "iter": iteration,
                                    "encoder": copy.deepcopy(speaker.encoder),
                                    "encoder_optm": copy.deepcopy(
                                        speaker.encoder_optimizer
                                    ),
                                    "decoder": copy.deepcopy(speaker.decoder),
                                    "decoder_optm": copy.deepcopy(
                                        speaker.decoder_optimizer
                                    ),
                                }

                    # Screen print out
                    logger.info("Bleu Score: %0.4f " % (bleu_score))
                    logger.info(
                        "Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f"
                        % tuple(precisions)
                    )

            logger.info(
                "PROGRESS: {}%".format(float(iteration) / args.num_iterations * 100)
            )
            logger.info("EVALERR: {}%".format(bleu_unseen * 100))

            # Save results
            df = pd.DataFrame(data_log)
            df.set_index("iteration")
            df_path = args.speaker_results_dir + "/log.csv"
            df.to_csv(df_path)
            save(
                best_model,
                best_iter,
                iteration,
                args.speaker_results_dir + "/best_val_unseen",
            )


def train(args, features):
    vocab = read_vocab(args.train_vocab)
    tok = Tokenizer(vocab=vocab, encoding_length=args.max_seq_length)

    train_dataset = R2RDataset(
        splits=["train"],
        tokenizer=tok,
        path_type=args.path_type,
        history=args.history,
    )

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank in [-2, -1]
        else DistributedSampler(train_dataset)
    )

    train_data_loader = R2RDataLoader(
        dataset=train_dataset,
        feature_store=features,
        batch_size=args.train_batch_size,
        collate_fn=R2RDataLoader_collate_fn,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_splits = ["val_seen", "val_unseen"]

    val_envs = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if args.local_rank in [-2, -1, 0]:
        # Create validation environments
        for split in val_splits:
            val_dataset = R2RDataset(
                splits=[split],
                tokenizer=tok,
                path_type=args.path_type,
                history=args.history,
            )
            evaluator = Evaluation(
                [split],
                path_type=args.path_type,
                results_dir=args.agent_results_dir,
                steps_to_next_q=args.steps_to_next_q,
            )

            val_sampler = SequentialSampler(val_dataset)
            val_data_loader = R2RDataLoader(
                dataset=val_dataset,
                feature_store=features,
                batch_size=args.eval_batch_size,
                collate_fn=R2RDataLoader_collate_fn,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            val_envs[split] = (val_data_loader, evaluator)

    encoder_path = os.path.join(args.model_name_or_path, "encoder")
    decoder_path = os.path.join(args.model_name_or_path, "decoder")
    tokenizer_path = args.model_name_or_path
    oscar_tokenizer = BertTokenizer.from_pretrained(
        tokenizer_path,
        do_lower_case=True,
    )

    tmp_root_folder = "srv/oscar_pretrained_models/base-vg-labels/ep_107_1192087"
    config_path = os.path.join(tmp_root_folder, "config.json")

    config = BertConfig.from_pretrained(config_path)

    config.img_feature_dim = args.img_feature_dim
    config.hidden_dropout_prob = args.drop_out
    config.classifier = "linear"
    config.loss_type = "CrossEntropy"
    config.cls_hidden_scale = 2

    config.action_space = args.action_space

    config.detector_classes = len(load_detector_classes())

    add_new_extra_embeds = not args.oscar_setting

    if add_new_extra_embeds:
        config.vocab_size = config.vocab_size + 3
        config.special_vocab_size = config.vocab_size
        config.type_vocab_size = config.type_vocab_size + 4
        config.max_position_embeddings = args.max_seq_length

    else:
        config.special_vocab_size = config.vocab_size

    model_class = MODEL_CLASS["PreTrainOscar"][1]
    oscar_model = model_class(config)
    bert_encoder = oscar_model.bert

    encoder = model.OscarEncoder(
        args=args,
        bert=bert_encoder,
        hidden_size=args.encoder_hidden_size,
        decoder_hidden_size=args.rnn_dim,
        dropout_ratio=args.dropout,
        bidirectional=args.bidir,
    ).to(args.device)

    decoder = model.AttnDecoderLSTMwithClassifier(
        args.angle_feat_size,
        args.aemb,
        args.rnn_dim,
        args.dropout,
        feature_size=args.lstm_img_feature_dim + args.angle_feat_size,
    ).to(args.device)

    if args.n_gpu > 1:
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(decoder)

    if args.local_rank not in [-2, -1]:
        encoder = torch.nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        decoder = torch.nn.parallel.DistributedDataParallel(
            decoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    encoder_optimizer = optim.Adam(
        encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    critic = model.Critic(args).to(args.device)
    if args.n_gpu > 1:
        critic = torch.nn.DataParallel(critic)
    if args.local_rank not in [-2, -1]:
        critic = torch.nn.parallel.DistributedDataParallel(
            critic,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    critic_optimizer = optim.Adam(
        critic.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    agent = Agent(
        args=args,
        dataloader=train_data_loader,
        results_path="",
        encoder=encoder,
        encoder_optimizer=encoder_optimizer,
        decoder=decoder,
        decoder_optimizer=decoder_optimizer,
        train_episode_len=args.max_episode_len,
        eval_episode_len=args.max_episode_len,
        turn_based=True,
        current_q_a_only=False,
        critic=critic,
        critic_optimizer=critic_optimizer,
        use_rl=True,
        agent_rl=True,
        random_start=False,
        J=False,
        steps_to_next_q=4,
        train_branching=1,
        action_probs_branching=False,
        oscar_tokenizer=oscar_tokenizer,
    )

    speaker = Speaker(
        args=args,
        dataloader=train_data_loader,
        agent=agent,
        feedback=args.speaker_feedback_method,
        maxDecode=args.max_seq_length,
        speaker_rl=True,
    )
    agent.speaker = speaker
    agent.speaker.asker_oracle = agent.speaker
    agent.speaker.helper_agent = agent
    speaker_best_model = {
        "iter": 0,
        "encoder": copy.deepcopy(speaker.encoder),
        "encoder_optm": copy.deepcopy(speaker.encoder_optimizer),
        "decoder": copy.deepcopy(speaker.decoder),
        "decoder_optm": copy.deepcopy(speaker.decoder_optimizer),
    }
    # Set up to save best model
    best_iter, start_iter, speaker_best_iter = 0, 0, 0
    best_bleu = defaultdict(lambda: 0)
    best_val = {
        "val_seen": {"goal_progress": 0.0},
        "val_unseen": {"goal_progress": 0.0},
    }
    output_metrics = [
        "success_rate",
        "oracle success_rate",
        "oracle path_success_rate",
        "dist_to_end_reduction",
        "ceiling",
    ]
    best_model = {
        "iter": 0,
        "encoder": copy.deepcopy(agent.encoder),
        "encoder_optm": copy.deepcopy(agent.encoder_optimizer),
        "decoder": copy.deepcopy(agent.decoder),
        "decoder_optm": copy.deepcopy(agent.decoder_optimizer),
    }
    best_model["critic"] = copy.deepcopy(agent.critic)
    best_model["critic_optm"] = copy.deepcopy(agent.critic_optimizer)

    data_log, speaker_data_log = defaultdict(list), defaultdict(list)

    agent.load(encoder_path, decoder_path)
    agent.speaker.load(args.saved_speaker_model_file)

    # Run train-eval loop
    start = time.time()
    for idx in range(start_iter, args.num_iterations, args.logging_steps):
        interval = min(args.logging_steps, args.num_iterations - idx)
        iteration = idx + interval

        print("EPOCH %d" % idx)

        # Train for log_every interval
        if not args.eval_only:
            agent.train(interval, feedback=args.agent_feedback_method)
            train_losses = np.array(agent.losses)
            assert len(train_losses) == interval
            train_loss_avg = np.average(train_losses)
            data_log["train loss"].append(train_loss_avg)
            loss_str = "train loss: %.4f" % train_loss_avg
            data_log["iteration"].append(iteration)
            speaker_data_log["iteration"].append(iteration)
        else:
            loss_str = ""
        goal_progress, ceiling = 0, 10  # avg

        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            agent.dataloader = env
            print("............ Evaluating %s ............." % env_name)
            agent.results_path = (
                args.agent_results_dir + "/iter_%d_%s.json" % (iteration, env_name)
                if not args.eval_only
                else args.agent_results_dir + "/eval.json"
            )
            # Get validation loss under the same conditions as training
            agent.test(
                use_dropout=True, feedback=args.agent_feedback_method, allow_cheat=True
            )
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log["%s loss" % env_name].append(val_loss_avg)
            # Get validation distance from goal under test evaluation conditions
            agent.test(use_dropout=False, feedback="argmax")
            agent.write_results()
            score_summary, _, gps = evaluator.score(agent.results_path)
            loss_str += ", %s loss: %.4f" % (env_name, val_loss_avg)
            for metric, val in score_summary.items():
                data_log["%s %s" % (env_name, metric)].append(val)
                if metric in output_metrics:
                    loss_str += ", %s: %.3f" % (metric, val)

                    # Save model according to goal_progress
                    if (
                        metric == "dist_to_end_reduction"
                        and val > best_val[env_name]["goal_progress"]
                    ):
                        best_val[env_name]["goal_progress"] = val
                        if env_name == "val_unseen":
                            goal_progress = 0 if val < 0 else val
                            best_iter = iteration
                            best_model = {
                                "iter": 0,
                                "encoder": copy.deepcopy(agent.encoder),
                                "encoder_optm": copy.deepcopy(agent.encoder_optimizer),
                                "decoder": copy.deepcopy(agent.decoder),
                                "decoder_optm": copy.deepcopy(agent.decoder_optimizer),
                            }
                            if args.agent_rl:
                                best_model["critic"] = copy.deepcopy(agent.critic)
                                best_model["critic_optm"] = copy.deepcopy(
                                    agent.critic_optimizer
                                )
                            if metric == "ceiling":
                                ceiling = val
            if args.agent_with_speaker:
                generated_dialog = agent.get_generated_dialog()
                path_id = random.choice(generated_dialog.keys())
                print("\n\n")
                print("Inference: ", generated_dialog[path_id])
                print("GT: ", evaluator.gt[path_id]["dialog_history"])
                if len(dialog_to_string(generated_dialog[path_id])) == 0:
                    continue
                bleu_score, precisions = evaluator.bleu_score(
                    generated_dialog, tok=tok, use_dialog_history=True
                )
                # Screen print out
                print("Bleu Score: %0.4f " % (bleu_score))
                print(
                    "Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f"
                    % tuple(precisions)
                )
                speaker_data_log["%s bleu" % (env_name)].append(bleu_score)
                # speaker_data_log['%s loss' % (env_name)].append(loss)

                # Save the model according to the bleu score
                if bleu_score > best_bleu[env_name]:
                    best_bleu[env_name] = bleu_score
                    if env_name == "val_unseen":
                        print(
                            "Save the model with %s BEST env bleu %0.4f"
                            % (env_name, bleu_score)
                        )
                        # bleu_unseen = bleu_score
                        speaker_best_iter = iteration
                        speaker_best_model = {
                            "iter": iteration,
                            "encoder": copy.deepcopy(speaker.encoder),
                            "encoder_optm": copy.deepcopy(speaker.encoder_optimizer),
                            "decoder": copy.deepcopy(speaker.decoder),
                            "decoder_optm": copy.deepcopy(speaker.decoder_optimizer),
                        }
            if not args.eval_only:
                pd.DataFrame(gps).T.to_csv(
                    args.agent_results_dir + "/" + env_name + "_gps.csv"
                )
            else:
                pd.DataFrame(gps).T.to_csv(
                    args.agent_results_dir + "/" + env_name + "_eval_gps.csv"
                )
        agent.dataloader = train_data_loader

        logger.info(
            "%s (%d %d%%) %s"
            % (
                timeSince(start, float(iteration) / args.num_iterations),
                iteration,
                float(iteration) / args.num_iterations * 100,
                loss_str,
            )
        )

        # Save results
        if not args.eval_only:
            df = pd.DataFrame(data_log)
            df.set_index("iteration")
            df_path = args.agent_results_dir + "/log.csv"
            df.to_csv(df_path)
            save(
                best_model,
                best_iter,
                iteration,
                args.agent_results_dir + "/best_val_unseen",
                with_critic=args.agent_rl,
            )

            if args.agent_with_speaker:
                # Save results
                df = pd.DataFrame(speaker_data_log)
                df.set_index("iteration")
                df_path = args.speaker_results_dir + "/log.csv"
                df.to_csv(df_path)
                save(
                    speaker_best_model,
                    speaker_best_iter,
                    iteration,
                    args.speaker_results_dir + "/best_val_unseen",
                )


def main():

    if (
        args.local_rank in [-2, -1, 0]
        and os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.eval_only
        and not args.test_only
        and not args.debug
    ):
        raise IOError(
            "%s \nOutput Directory not empty and train setting is on. Exiting to prevent overwriting..."
            % (args.output_dir)
        )

    if not os.path.exists(args.output_dir) and args.local_rank in [-2, -1, 0]:
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"))
        os.makedirs(os.path.join(args.output_dir, "predictions"))
        os.makedirs(os.path.join(args.output_dir, "results"))
        os.makedirs(os.path.join(args.output_dir, "tensorboard"))
        os.makedirs(args.speaker_results_dir)
        os.makedirs(args.agent_results_dir)

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

    if args.path_type == "planner_path":
        args.max_episode_len = 30
    else:
        args.max_episode_len = 80

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank not in [-2, -1]),
        "False",
    )

    # Set seed
    set_seed(args.seed, args.n_gpu)

    setup_vocab(args.train_vocab, args.trainval_vocab)

    logger.info("Training/evaluation parameters %s", args)

    if args.debug:
        feature_path = None
    else:
        feature_path = os.path.join(args.img_feat_dir, args.img_feature_file)
    features = read_tsv_img_features(
        path=feature_path,
        feature_size=args.lstm_img_feature_dim,
    )

    if args.speaker_only:
        train_speaker(args, features)
    else:
        if args.eval_only:
            assert (
                len(args.eval_iters) != 0 and args.eval_iters != -1
            ), "incorrect eval_iters provided!"
            val(args, features, args.eval_iters)
        else:
            train(args, features)

    sys.exit()


if __name__ == "__main__":
    main()
