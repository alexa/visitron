import copy
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

# from torch.autograd import Variable
from tqdm import tqdm

from agent import Seq2SeqAgent
from env import R2RBatch
from eval import Evaluation
from model import AttnDecoderLSTM, Critic, EncoderLSTM
from params import args
from speaker import Speaker
from utils import (
    GPTTokenizer,
    Tokenizer,
    build_vocab,
    dialog_to_string,
    load,
    load_features,
    padding_idx,
    read_vocab,
    save,
    timeSince,
    write_vocab,
)

warnings.filterwarnings("ignore")


print(args)  # Print finalized args

""" Train on training set, validating on both seen and unseen. """


def train(train_env, agent, val_envs={}, tok=None):

    # Load speaker
    speaker_best_model = None
    if args.agent_with_speaker:
        speaker = Speaker(
            train_env,
            agent,
            feedback=args.speaker_feedback_method,
            maxDecode=args.MAX_INPUT_LENGTH,
            speaker_rl=args.speaker_rl,
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
        if not os.path.isdir(args.speaker_results_dir):
            os.makedirs(args.speaker_results_dir)

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
    if args.agent_rl:
        best_model["critic"] = copy.deepcopy(agent.critic)
        best_model["critic_optm"] = copy.deepcopy(agent.critic_optimizer)
    data_log, speaker_data_log = defaultdict(list), defaultdict(list)
    if not os.path.isdir(args.agent_results_dir):
        os.makedirs(args.agent_results_dir)

    # Load previous partial work or agent, speaker pre-training
    if args.load_agent:
        # print("Wanted file: %s" % args.saved_agent_model_file)
        if args.rw_results and os.path.isfile(args.saved_agent_model_file):
            start_iter = load(
                best_model, args.saved_agent_model_file, with_critic=args.agent_rl
            )
            print(
                "Load the agent's state dict from %s at iteration %d"
                % (args.saved_agent_model_file, start_iter)
            )
        elif os.path.isfile(args.agent_pretrain_file):
            prev_iter = load(best_model, args.agent_pretrain_file)
            print(
                "Load the agent's state dict from %s at iteration %d"
                % (args.agent_pretrain_file, prev_iter)
            )
    if args.load_speaker:
        if args.rw_results and os.path.isfile(args.saved_speaker_model_file):
            agent.speaker.load(args.saved_speaker_model_file)
            if not args.eval_only and os.path.isfile(
                args.in_speaker_results_dir + "/log.csv"
            ):
                speaker_data_log = pd.read_csv(
                    args.in_speaker_results_dir + "/log.csv", index_col=0
                ).to_dict("l")
                speaker_data_log = defaultdict(list, speaker_data_log)
        elif os.path.isfile(args.speaker_pretrain_file):
            agent.speaker.load(args.speaker_pretrain_file)
    if (
        not args.eval_only
        and args.rw_results
        and os.path.isfile(args.in_agent_results_dir + "/log.csv")
    ):
        data_log = pd.read_csv(
            args.in_agent_results_dir + "/log.csv", index_col=0
        ).to_dict("l")
        data_log = defaultdict(list, data_log)

    # Run train-eval loop
    start = time.time()
    for idx in range(start_iter, args.N_ITERS, args.log_every):
        interval = min(args.log_every, args.N_ITERS - idx)
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
            agent.env = env
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
        agent.env = train_env

        # Screen print out
        if args.philly:
            print("PROGRESS: {}%".format(float(iteration) / args.N_ITERS * 100))
            print("EVALERR: {}%".format(goal_progress / ceiling * 100))
        print(
            "%s (%d %d%%) %s"
            % (
                timeSince(start, float(iteration) / args.N_ITERS),
                iteration,
                float(iteration) / args.N_ITERS * 100,
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


def train_speaker(train_env, agent, val_envs=None, tok=None):
    speaker = Speaker(
        train_env,
        agent,
        decoder=args.speaker_decoder,
        feedback=args.speaker_feedback_method,
        maxDecode=args.MAX_INPUT_LENGTH,
        speaker_rl=args.speaker_rl,
        tokenizer=tok,
    )

    # Set up to save best model
    start_iter, best_iter = 0, 0
    best_bleu = defaultdict(lambda: 0)
    # best_loss = defaultdict(lambda: 1232)
    data_log = defaultdict(list)
    best_model = {
        "iter": 0,
        "encoder": copy.deepcopy(speaker.encoder),
        "encoder_optm": copy.deepcopy(speaker.encoder_optimizer),
        "decoder": copy.deepcopy(speaker.decoder),
        "decoder_optm": copy.deepcopy(speaker.decoder_optimizer),
    }
    if not os.path.isdir(args.speaker_results_dir):
        os.makedirs(args.speaker_results_dir)

    # Load previous partial work or pre-train
    if args.rw_results and os.path.isfile(args.saved_speaker_model_file):
        start_iter = speaker.load(args.saved_speaker_model_file)
        if os.path.isfile(args.in_speaker_results_dir + "/log.csv"):
            data_log = pd.read_csv(
                args.in_speaker_results_dir + "/log.csv", index_col=0
            ).to_dict("l")
            data_log = defaultdict(list, data_log)
    elif args.load_speaker and os.path.isfile(args.speaker_pretrain_file):
        speaker.load(args.speaker_pretrain_file)

    # Run train-eval loop
    for idx in tqdm(
        range(start_iter, args.N_ITERS, args.log_every), desc="Train-Eval Loop"
    ):
        interval = min(args.log_every, args.N_ITERS - idx)
        iteration = idx + interval
        data_log["iteration"].append(iteration)
        bleu_unseen = 0

        # Train for log_every interval
        speaker.env = train_env
        speaker.train(1, train_nav=True)  # Train nav
        speaker.train(interval)  # Train ora

        print("Iter: %d" % idx)

        # Evaluation
        for env_name, (env, evaluator) in val_envs.items():
            if "train" in env_name:  # Ignore the large training set for the efficiency
                continue
            print("............ Evaluating %s ............." % env_name)
            speaker.env = env
            for instr_type in ["nav", "ora"]:
                for_nav = instr_type == "nav"
                print("............ %s ............." % instr_type)
                path2inst, loss, word_accu, sent_accu = speaker.valid(for_nav=for_nav)
                path_id = next(iter(path2inst.keys()))
                print("\n\n")
                print("Inference: ", tok.decode_sentence(path2inst[path_id]))
                print("GT: ", evaluator.gt[path_id][instr_type + "_instructions"])
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
                            print(
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
                print("Bleu Score: %0.4f " % (bleu_score))
                print(
                    "Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f"
                    % tuple(precisions)
                )

        print("PROGRESS: {}%".format(float(iteration) / args.N_ITERS * 100))
        print("EVALERR: {}%".format(bleu_unseen * 100))

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


def setup():
    torch.manual_seed(1)
    if args.device == "cuda":
        torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(args.train_vocab):
        write_vocab(build_vocab(splits=["train"]), args.train_vocab)
    if not os.path.exists(args.trainval_vocab):
        write_vocab(
            build_vocab(splits=["train", "val_seen", "val_unseen"]), args.trainval_vocab
        )


def test_submission():
    """ Train on combined training and validation sets, and generate test submission. """

    setup()

    # Model prefix to uniquely id this instance.
    model_prefix = "%s-seq2seq-%s-%s-%d-%s-imagenet" % (
        args.eval_type,
        args.history,
        args.path_type,
        args.MAX_EPISODE_LEN,
        args.agent_feedback_method,
    )
    if args.blind:
        model_prefix += "-blind"

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(args.trainval_vocab)
    tok = Tokenizer(vocab=vocab, encoding_length=args.MAX_INPUT_LENGTH)
    feats, feats_info = load_features(args.FEATURES, args.blind)
    train_env = R2RBatch(
        feats,
        feats_info,
        batch_size=args.BATCH_SIZE,
        splits=["train", "val_seen", "val_unseen"],
        tokenizer=tok,
        path_type=args.path_type,
        history=args.history,
        blind=args.blind,
    )

    # Build models and train
    enc_hidden_size = args.HIDDEN_SIZE // 2 if args.BIDIRECTIONAL else args.HIDDEN_SIZE
    encoder = EncoderLSTM(
        len(vocab),
        args.WORD_EMBEDDING_SIZE,
        enc_hidden_size,
        padding_idx,
        args.DROPOUT_RATIO,
        bidirectional=args.BIDIRECTIONAL,
    ).to(args.device)
    decoder = AttnDecoderLSTM(
        Seq2SeqAgent.n_inputs(),
        Seq2SeqAgent.n_outputs(),
        args.ACTION_EMBEDDING_SIZE,
        args.HIDDEN_SIZE,
        args.DROPOUT_RATIO,
    ).to(args.device)
    train(
        train_env, encoder, decoder, args.path_type, args.MAX_INPUT_LENGTH, model_prefix
    )

    # Generate test submission
    test_env = R2RBatch(
        feats,
        feats_info,
        batch_size=args.BATCH_SIZE,
        splits=["test"],
        tokenizer=tok,
        path_type=args.path_type,
        history=args.history,
        blind=args.blind,
    )
    agent = Seq2SeqAgent(test_env, "", encoder, decoder, args.MAX_EPISODE_LEN)
    agent.results_path = "%s/%s_%s_iter_%d.json" % (
        args.results_dir,
        model_prefix,
        "test",
        20000,
    )
    agent.test(use_dropout=False, feedback="argmax")
    agent.write_results()


def train_val(use_test_split=False):
    """ Train on the training set, and validate on seen and unseen splits. """

    setup()
    train_splits = (
        ["train"] if not use_test_split else ["train", "val_seen", "val_unseen"]
    )
    val_splits = ["val_seen", "val_unseen"] if not use_test_split else ["test"]

    # Create a batch training environment that will also preprocess text
    if args.speaker_decoder == "gpt2":
        tok = GPTTokenizer(args.MAX_INPUT_LENGTH)
    else:
        vocab = read_vocab(args.train_vocab)
        tok = Tokenizer(vocab=vocab, encoding_length=args.MAX_INPUT_LENGTH)
    feats, feats_info = load_features(args.FEATURES, args.blind, debug=args.debug)
    train_env = R2RBatch(
        feats,
        feats_info,
        batch_size=args.BATCH_SIZE,
        splits=train_splits,
        tokenizer=tok,
        path_type=args.path_type,
        history=args.history,
        blind=args.blind,
        datasets=args.train_datasets,
        mount_dir=args.mount_dir,
        segmented=args.segmented,
        speaker_only=args.speaker_only,
    )

    # Create validation environments
    val_envs = {
        split: (
            R2RBatch(
                feats,
                feats_info,
                batch_size=args.BATCH_SIZE,
                splits=[split],
                tokenizer=tok,
                path_type=args.path_type,
                history=args.history,
                blind=args.blind,
                datasets=args.eval_datasets,
                mount_dir=args.mount_dir,
                segmented=args.segmented,
                speaker_only=args.speaker_only,
            ),
            Evaluation(
                [split],
                path_type=args.path_type,
                datasets=args.eval_datasets,
                mount_dir=args.mount_dir,
                segmented=args.segmented,
                speaker_only=args.speaker_only,
                results_dir=args.agent_results_dir,
                steps_to_next_q=args.steps_to_next_q,
            ),
        )
        for split in val_splits
    }

    # Build models and train
    enc_hidden_size = args.HIDDEN_SIZE // 2 if args.BIDIRECTIONAL else args.HIDDEN_SIZE
    encoder = EncoderLSTM(
        tok.vocab_size(),
        args.WORD_EMBEDDING_SIZE,
        enc_hidden_size,
        padding_idx,
        args.DROPOUT_RATIO,
        bidirectional=args.BIDIRECTIONAL,
    ).to(args.device)
    decoder = AttnDecoderLSTM(
        Seq2SeqAgent.n_inputs(),
        Seq2SeqAgent.n_outputs(),
        args.ACTION_EMBEDDING_SIZE,
        args.HIDDEN_SIZE,
        args.DROPOUT_RATIO,
    ).to(args.device)
    encoder_optimizer = optim.Adam(
        encoder.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY
    )
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY
    )

    if args.parallelize:
        encoder = nn.DataParallel(encoder).to(args.device)
        decoder = nn.DataParallel(decoder).to(args.device)

    # Create the follower agent
    agent = None
    if args.AGENT_TYPE == "seq2seq":
        critic = None if not args.agent_rl else Critic().to(args.device)
        critic_optimizer = (
            None
            if not args.agent_rl
            else optim.Adam(
                critic.parameters(),
                lr=args.LEARNING_RATE,
                weight_decay=args.WEIGHT_DECAY,
            )
        )
        agent = Seq2SeqAgent(
            train_env,
            "",
            encoder,
            encoder_optimizer,
            decoder,
            decoder_optimizer,
            train_episode_len=args.train_max_steps,
            eval_episode_len=args.eval_max_steps,
            turn_based=args.turn_based_agent,
            current_q_a_only=args.current_q_a_only,
            critic=critic,
            critic_optimizer=critic_optimizer,
            use_rl=args.use_rl,
            agent_rl=args.agent_rl,
            random_start=args.random_start,
            J=args.J,
            steps_to_next_q=args.steps_to_next_q,
            train_branching=args.train_branching,
            eval_branching=args.eval_branching,
            action_probs_branching=args.action_probs_branching,
        )
    else:
        sys.exit("Unrecognized agent_type '%s'" % args.AGENT_TYPE)

    if args.speaker_only:
        print("Training a speaker with %s feedback" % (args.speaker_feedback_method))
        train_speaker(train_env, agent, val_envs, tok)
    else:
        print(
            "Training a %s agent with %s feedback"
            % (args.AGENT_TYPE, args.agent_feedback_method)
        )
        train(train_env, agent, val_envs, tok)


if __name__ == "__main__":
    if args.eval_type == "val":
        train_val()
    else:
        train_val(use_test_split=True)
    # test_submission()
