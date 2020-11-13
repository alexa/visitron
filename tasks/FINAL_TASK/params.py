# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse

parser = argparse.ArgumentParser()

## ## Required parameters

## Directories parameters
parser.add_argument(
    "--img_feat_dir",
    default=None,
    type=str,
    required=True,
    help="Image features directory",
)
parser.add_argument(
    "--img_feature_file",
    default=None,
    type=str,
    required=True,
    help="Image features file",
)
parser.add_argument(
    "--candidate_feature_file",
    default=None,
    type=str,
    required=False,
    help="Candidate Image features file",
)
parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    required=True,
    help="Data related to task such CVDN or NDH dataset",
)
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=False,
    help="Path to pre-trained model or shortcut name",
)
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="Directory path to save results and tensorboard logs",
)

parser.add_argument(
    "--agent",
    default="oscar",
    type=str,
    choices=["oscar", "att-lstm"],
    help="Which agent to use",
)
parser.add_argument(
    "--no_pretrained_model",
    default=False,
    action="store_true",
    help="Use no pretrained weights for Oscar encoder",
)

parser.add_argument(
    "--tar_back",
    default=False,
    action="store_true",
    help="Use [TAR] after dialog",
)

parser.add_argument(
    '--no_oscar_setting',
    dest='oscar_setting',
    action='store_false'
)

parser.add_argument(
    "--add_ndh_data",
    default=True,
    action="store_true",
    help="Use NDH data",
)
parser.add_argument(
    "--add_r2r_data",
    default=False,
    action="store_true",
    help="Add interleaving of R2R",
)
parser.add_argument(
    "--add_r4r_data",
    default=False,
    action="store_true",
    help="Add interleaving of R4R",
)
parser.add_argument(
    "--add_rxr_data",
    default=False,
    action="store_true",
    help="Add interleaving of RxR",
)

parser.add_argument(
    "--encoder",
    default="lstm",
    type=str,
    choices=["lstm", "oscar"],
    help="Which encoder to use",
)

## Input parameters
parser.add_argument(
    "--max_seq_length",
    default=512,
    type=int,
    help="The maximum total input sequence length on the text side after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--max_img_seq_length",
    default=256,
    type=int,
    help="The maximum total input image sequence length.",
)
parser.add_argument(
    "--angle_feat_size",
    default=4,
    type=int,
    help="Size of angle feature",
)
parser.add_argument(
    "--action_space",
    default=36,
    type=int,
    help="Number of actions",
)
parser.add_argument(
    "--ignoreid",
    default=-100,
    type=int,
    help="Ignore id",
)
parser.add_argument(
    "--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension."
)
parser.add_argument(
    "--lstm_img_feature_dim",
    default=2054,
    type=int,
    help="The Image Feature Dimension.",
)
parser.add_argument(
    "--word_embed_size", default=768, type=int, help="The Image Feature Dimension."
)
parser.add_argument("--views", default=36, type=int, help="Views.")
parser.add_argument("--bidir", action="store_true", default=False)
parser.add_argument("--encoder_hidden_size", type=int, default=512)
parser.add_argument("--rnnDim", dest="rnn_dim", type=int, default=512)
parser.add_argument("--aemb", type=int, default=64)
parser.add_argument(
    "--add_action_stream",
    default=False,
    action="store_true",
    help="Add action or not",
)
parser.add_argument(
    "--submit",
    default=False,
    action="store_true",
    help="Submit or not",
)

## Training setting parameters
parser.add_argument(
    "--path_type",
    default="trusted_path",
    type=str,
    choices=["planner_path", "player_path", "trusted_path"],
    help="Which path to use to provide supervision",
)
parser.add_argument(
    "--feedback_method",
    default="sample",
    type=str,
    choices=["sample", "teacher"],
    help="Teacher forcing or Student forcing",
)
parser.add_argument(
    "--detach_loss",
    action="store_true",
    help="Detach loss and compute gradients based on it inside the timestep loop",
)
parser.add_argument(
    "--detach_loss_at",
    default=20,
    type=int,
    help="Detach loss and compute gradients after X timesteps",
)
parser.add_argument(
    "--pretrained_fixed",
    action="store_true",
    help="Only finetune the final layer",
)
parser.add_argument(
    "--train_only",
    action="store_true",
    help="Only train the model, no eval",
)
parser.add_argument(
    "--eval_only",
    action="store_true",
    help="Only eval the model, no training",
)
parser.add_argument(
    "--test_only",
    action="store_true",
    help="Test submission for EvalAI",
)
parser.add_argument(
    "--eval_iters",
    default=-1,
    type=int,
    nargs="*",
    help="Use list of eval_iter iteration model to evaluate model in eval_only setting",
)
parser.add_argument(
    "--evaluate_during_training",
    action="store_true",
    help="Run evaluation during training at each logging step.",
)
parser.add_argument(
    "--per_gpu_train_batch_size",
    default=16,
    type=int,
    help="Batch size per GPU for training.",
)
parser.add_argument(
    "--per_gpu_eval_batch_size",
    default=8,
    type=int,
    help="Batch size per GPU for evaluation.",
)
parser.add_argument(
    "--learning_rate",
    default=5e-5,
    type=float,
    help="The initial learning rate for Adam.",
)
parser.add_argument(
    "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
)
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
)
parser.add_argument(
    "--scheduler", default="linear", type=str, help="constant or linear."
)
parser.add_argument(
    "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
)
parser.add_argument(
    "--num_iterations",
    default=20000,
    type=int,
    help="Total number of training iterations to perform.",
)
parser.add_argument(
    "--num_epochs",
    default=10,
    type=int,
    help="Total number of training epochs to perform.",
)
parser.add_argument(
    "--warmup_steps",
    default=0,
    type=int,
    help="Linear warmup over warmup_steps iterations.",
)
parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out for BERT.")
parser.add_argument("--dropout", default=0.5, type=float, help="Drop out for BERT.")
parser.add_argument(
    "--mlm_probability",
    default=0.15,
    type=float,
    help="Probability for masked-LM training",
)

parser.add_argument(
    "--masked_token_prediction",
    action="store_true",
    help="Whether to do masked token prediction using token classes or not. Default: Masked LM for tokens",
)


## Logging parameters
parser.add_argument(
    "--logging_steps",
    type=int,
    default=50,
    help="Log training process every X updates steps.",
)
parser.add_argument(
    "--eval_logging_steps",
    type=int,
    default=1000,
    help="Log eval process while training every X updates steps.",
)
parser.add_argument(
    "--save_steps",
    type=int,
    default=-1,
    help="Save checkpoint every X updates steps.",
)

parser.add_argument(
    "--seed", type=int, default=42, help="random seed for initialization"
)
parser.add_argument(
    "-j",
    "--num_workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="For distributed training: local_rank",
)

parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Use this arg for debug purposes",
)

parser.add_argument(
    "--slurm_info",
    default="",
    type=str,
    help="SLURM info to print",
)

args = parser.parse_args()
