import argparse
import os

import torch

parser = argparse.ArgumentParser()
# Source directories settings
parser.add_argument(
    "--train_vocab", type=str, default="tasks/CVDN/data/train_vocab.txt"
)
parser.add_argument(
    "--trainval_vocab", type=str, default="tasks/CVDN/data/trainval_vocab.txt"
)
parser.add_argument(
    "--imagenet_features", type=str, default="img_features/ResNet-152-imagenet.tsv"
)
parser.add_argument(
    "--train_datasets",
    type=str,
    default="CVDN_NDH",
    help="underscore separated training list",
)
parser.add_argument(
    "--eval_datasets", type=str, default="CVDN", help="underscore separated eval list"
)
parser.add_argument("--mount_dir", type=str, default="", help="data mount directory")
parser.add_argument(
    "--out_mount_dir", type=str, default="", help="output mount directory"
)
parser.add_argument("--combined_dir", type=str, default="")
parser.add_argument(
    "--pretrain_dir",
    type=str,
    default="rmm_results/baseline/CVDN_train_eval_CVDN/G1/v1/steps_4",
)  # NDH_R2R_train_eval_NDH')
parser.add_argument("--ver", type=str, default="v1")

# Model settings
parser.add_argument(
    "--mode", type=str, default="baseline", help="baseline, gameplay, or other"
)
parser.add_argument(
    "--entity", type=str, default="agent", help="agent, speaker, combined"
)
parser.add_argument(
    "--path_type",
    type=str,
    default="player_path",
    help="planner_path, player_path, or trusted_path",
)
parser.add_argument(
    "--history",
    type=str,
    default="all",
    help="none, target, oracle_ans, nav_q_oracle_ans, or all",
)
parser.add_argument("--eval_type", type=str, default="val", help="val or test")
parser.add_argument(
    "--blind",
    action="store_true",
    required=False,
    help="whether to replace the ResNet encodings with zero vectors at inference time",
)
parser.add_argument("--segmented", action="store_true", required=False)
parser.add_argument("--target_only", action="store_true", required=False)
parser.add_argument("--current_q_a_only", action="store_true", required=False)
parser.add_argument(
    "--rl_mode", type=str, default="", help="agent, speaker, agent_and_speaker"
)
parser.add_argument("--use_rl", action="store_true", required=False)
parser.add_argument("--random_start", action="store_true", required=False)
parser.add_argument("--J", type=int, default=0, help="Jitter")
parser.add_argument(
    "--steps_to_next_q",
    type=int,
    default=4,
    help="How many steps the follower takes before asking a question",
)
parser.add_argument("--train_branching", type=int, default=3, help="branching factor")
parser.add_argument(
    "--eval_branching", type=int, default=1, help="eval branching factor"
)
parser.add_argument(
    "--train_max_steps",
    type=int,
    default=80,
    help="Max number of steps during training",
)
parser.add_argument(
    "--eval_max_steps", type=int, default=80, help="Max number of steps during eval"
)
parser.add_argument("--action_probs_branching", action="store_true", required=False)
parser.add_argument("--eval_only", action="store_true", required=False)

# Agent settings
parser.add_argument(
    "--agent_feedback_method",
    type=str,
    default="sample",
    help="teacher, argmax, sample, topk, nucleus, temperature, penalty, nucleus_with_penalty",
)
parser.add_argument("--turn_based_agent", action="store_true", required=False)
parser.add_argument("--load_agent", action="store_true", required=False)
parser.add_argument("--agent_with_speaker", action="store_true", required=False)
parser.add_argument("--action_sampling", action="store_true", required=False)
parser.add_argument("--agent_rl", action="store_true", required=False)

# Speaker settings
parser.add_argument(
    "--speaker_feedback_method",
    type=str,
    default="sample",
    help="teacher, argmax, sample, topk, nucleus, temperature, penalty, nucleus_with_penalty",
)
parser.add_argument("--speaker_only", action="store_true", required=False)
parser.add_argument("--load_speaker", action="store_true", required=False)
parser.add_argument("--speaker_with_asker", action="store_true", required=False)
parser.add_argument("--speaker_rl", action="store_true", required=False)
parser.add_argument(
    "--speaker_decoder",
    type=str,
    default="lstm",
    choices=["lstm", "gpt2"],
    help="Which speaker decoder to use",
)

# Runtime settings
parser.add_argument("--log_every", type=int, default=100, help="how often to log")
parser.add_argument("--gpus", type=str, default="G1", help="G1, G2, G4, G8")
parser.add_argument("--philly", action="store_true", required=False)
parser.add_argument(
    "--rw_results",
    action="store_true",
    required=False,
    help="whether to allow partial saves and reloads",
)
parser.add_argument("--parallelize", action="store_true", required=False)

parser.add_argument("--local_rank", type=int, default=-1, help="Local Rank")
parser.add_argument("--debug", action="store_true", default=False)


args = parser.parse_args()
assert args.mode in ["baseline", "gameplay", "other"]

# On philly mnt point changes to /mnt/<container>
if args.philly:
    args.mount_dir = "/mnt/cvdn/"
    args.out_mount_dir = "/mnt/out_cvdn/"
    args.rw_results = True
    args.parallelize = True

args.train_vocab = args.mount_dir + args.train_vocab
args.trainval_vocab = args.mount_dir + args.trainval_vocab
args.imagenet_features = args.mount_dir + args.imagenet_features
args.pretrain_dir = args.mount_dir + args.pretrain_dir

# Setup results dirs for the agent and speaker
args.results_dir = os.path.join(
    "rmm_results",
    args.mode,
    args.train_datasets + "_train_eval_" + args.eval_datasets,
    args.gpus,
    args.ver,
)

if args.current_q_a_only:
    args.results_dir += "/current_t_q_a_only"
elif args.target_only:
    args.results_dir += "/target_only"

if args.random_start:
    args.results_dir += "/random_start_" + str(args.J)

if args.steps_to_next_q >= 0:
    args.results_dir += "/steps_" + str(args.steps_to_next_q)

# Adjust based on mode
if "agent" in args.rl_mode:
    args.use_rl = True
    args.agent_rl = True
    args.results_dir += "/agent_rl"
if "speaker" in args.rl_mode:
    args.use_rl = True
    args.speaker_rl = True
    prefix = "_" if "agent" in args.rl_mode else "/"
    args.results_dir += prefix + "speaker_rl"

if args.mode == "baseline":
    args.path_type = "trusted_path"
    if args.entity == "speaker":
        args.speaker_only = True
elif args.mode == "gameplay":
    args.history = "target"
    if args.target_only:
        args.path_type = "trusted_path"
    else:
        args.turn_based_agent = True
        args.path_type = "player_path"
        args.load_agent = True
        args.agent_with_speaker = True
        args.load_speaker = True
        if args.entity != "combined":
            args.single_turn = True
    args.results_dir += (
        "/agent_"
        + args.agent_feedback_method
        + "_speaker_"
        + args.speaker_feedback_method
    )

args.agent_dir = os.path.join(args.results_dir, "agent", args.agent_feedback_method)
args.speaker_dir = os.path.join(
    args.results_dir, "speaker", args.speaker_decoder, args.speaker_feedback_method
)
args.in_agent_results_dir = args.mount_dir + args.agent_dir
args.agent_results_dir = args.out_mount_dir + args.agent_dir
args.in_speaker_results_dir = args.mount_dir + args.speaker_dir
args.speaker_results_dir = args.out_mount_dir + args.speaker_dir
args.agent_pretrain_file = (
    args.pretrain_dir + "/" + os.path.join("agent", "sample", "best_val_unseen")
)
args.speaker_pretrain_file = (
    args.pretrain_dir + "/" + os.path.join("speaker", "argmax", "best_val_unseen")
)
args.saved_agent_model_file = args.mount_dir + args.agent_dir + "/best_val_unseen"
args.saved_speaker_model_file = args.mount_dir + args.speaker_dir + "/best_val_unseen"

# if args.mode in ['combined_subgoals', 'gameplay', 'full_gameplay', 'frase_full_gameplay', 'full_frase_full_gameplay', 'single_turn_full_frase_full_gameplay']:
#     args.turn_based_agent = True; args.path_type = 'player_path'
#     if 'gameplay' in args.mode:
#         args.load_agent = True
#         args.agent_with_speaker = True
#         args.load_speaker = True
#         args.speaker_pretrain_file = args.segmented_dir + '/speaker/best_val_unseen'
#         if 'full_gameplay' in args.mode:
#             args.agent_with_asker = True
#             if 'frase_full_gameplay' in args.mode:
#                 args.speaker_with_helper_agent = True
#                 if 'full_frase_full_gameplay' in args.mode:
#                     args.asker_with_oracle = True
#                     if args.mode = 'single_turn_full_frase_full_gameplay':
#                         args.single_turn = True
#     if args.speaker_only:
#         args.segmented = True
# else:
#     args.turn_based_agent = False
#     if args.mode == 'segmented':
#         args.segmented = True; args.path_type = 'player_path'
#         args.speaker_with_asker = True
#     else: #Baseline and pretrain
#         args.path_type = 'trusted_path'
#         if args.mode == 'action_sampling':
#             args.action_sampling = True

# Input settings.
# In MP, MAX_INPUT_LEN = 80 while average utt len is 29, e.g., a bit less than 3x avg.
# args.MAX_INPUT_LENGTH = 120 * 6  # 4.93+/-3.21 turns -> 2.465+/-1.605 Q/A. 5.67 at x2 std. Call it 6 (real max 13).
# if args.history == 'none':
#     args.MAX_INPUT_LENGTH = 1  # [<EOS>] fixed length.
# elif args.history == 'target':
#     args.MAX_INPUT_LENGTH = 3  # [<TAR> target <EOS>] fixed length.
# elif args.history == 'oracle_ans':
#     args.MAX_INPUT_LENGTH = 70  # 16.16+/-9.67 ora utt len, 35.5 at x2 stddevs. 71 is double that.
# elif args.history == 'nav_q_oracle_ans':
#     args.MAX_INPUT_LENGTH = 120  # 11.24+/-6.43 [plus Ora avg], 24.1 at x2 std. 71+48 ~~ 120 per QA doubles both.
# elif args.history == 'all':
#     args.MAX_INPUT_LENGTH = 160

# Training settings.
args.AGENT_TYPE = "seq2seq"
args.N_ITERS = 5000 if args.agent_feedback_method == "teacher" else 20000
# args.N_ITERS = args.N_ITERS if args.mode == 'baseline' else 10
args.MAX_EPISODE_LEN = (
    80 if args.path_type != "planner_path" else 20
)  # Heuristic from Jesse
args.MAX_EPISODE_LEN = args.MAX_EPISODE_LEN if not args.eval_only else 250
args.FEATURES = args.imagenet_features
args.BATCH_SIZE = 2
args.MAX_INPUT_LENGTH = 160  # 4.93+/-3.21 turns -> 2.465+/-1.605 Q/A. 5.67 at x2 std. Call it 6 (real max 13).
args.WORD_EMBEDDING_SIZE = 256
args.ACTION_EMBEDDING_SIZE = 32
args.TARGET_EMBEDDING_SIZE = 32
args.HIDDEN_SIZE = 512
args.BIDIRECTIONAL = False
args.DROPOUT_RATIO = 0.5
args.LEARNING_RATE = 0.0001
args.WEIGHT_DECAY = 0.0005

# Setup CPU, CUDA, GPU & distributed training
if args.local_rank == -1:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        if args.n_gpu > 1:
            assert False, "Data parallel training not setup!"
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    assert False, "Distributed training not setup!"
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
args.device = device

print(f"Using {args.device}")
