# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import copy
import logging
import math
import os
import random
import re
import string
import sys
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F

import utils
from utils_data import load_datasets

logger = logging.getLogger(__name__)


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


# padding, unknown word, end of sentence
base_vocab = ["<PAD>", "<UNK>", "<EOS>", "<NAV>", "<ORA>", "<TAR>"]
padding_idx = base_vocab.index("<PAD>")


def load(best_model, path, with_critic=False, parallel=True):
    """ Loads parameters (but not training state) """
    states = torch.load(path)

    def recover_state(name, model, optimizer):
        state = model.state_dict()
        state.update(states[name]["state_dict"])
        # ###########  Comment out if `module` prefix is required #############
        if parallel:
            new_state = {}
            for k, v in state.items():
                key = k[7:] if k[:7] == "module." else k
                new_state[key] = v
            state = new_state
        #######################################################################
        model.load_state_dict(state)
        optimizer.load_state_dict(states[name]["optimizer"])

    all_tuple = [
        ("encoder", best_model["encoder"], best_model["encoder_optm"]),
        ("decoder", best_model["decoder"], best_model["decoder_optm"]),
    ]
    if with_critic:
        all_tuple.append(("critic", best_model["critic"], best_model["critic_optm"]))
    for param in all_tuple:
        recover_state(*param)
    return states["encoder"]["iteration"]


def save(best_model, epoch, iteration, path, with_critic=False):
    """ Snapshot models """
    the_dir, _ = os.path.split(path)
    if not os.path.isdir(the_dir):
        os.makedirs(the_dir)
    states = {}

    def create_state(name, model, optimizer):
        states[name] = {
            "epoch": epoch + 1,
            "iteration": iteration,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

    all_tuple = [
        ("encoder", best_model["encoder"], best_model["encoder_optm"]),
        ("decoder", best_model["decoder"], best_model["decoder_optm"]),
    ]
    if with_critic:
        all_tuple.append(("critic", best_model["critic"], best_model["critic_optm"]))
    for param in all_tuple:
        create_state(*param)
    torch.save(states, path)


def build_vocab(splits=["train"], min_count=5, start_vocab=base_vocab):
    """ Build a vocab, starting with base vocab containing a few useful tokens. """
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits)
    for item in data:
        for turn in item["dialog_history"]:
            count.update(t.split_sentence(turn["message"]))
    vocab = list(start_vocab)

    # Add words that are object targets.
    targets = set()
    for item in data:
        target = item["target"]
        targets.add(target)
    vocab.extend(list(targets))

    # Add words above min_count threshold.
    for word, num in count.most_common():
        if word in vocab:  # targets strings may also appear as regular vocabulary.
            continue
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print("Writing vocab of size %d to %s" % (len(vocab), path))
    with open(path, "w") as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def setup_vocab(TRAIN_VOCAB, TRAINVAL_VOCAB):

    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=["train"]), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(
            build_vocab(splits=["train", "val_seen", "val_unseen"]), TRAINVAL_VOCAB
        )


class Tokenizer(object):
    """ Class to tokenize and encode a sentence. """

    SENTENCE_SPLIT_REGEX = re.compile(
        r"(\W+)"
    )  # Split on any non-alphanumeric character

    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self._word_to_index = {}
        self._index_to_word = {}
        if vocab:
            for i, word in enumerate(vocab):
                self._word_to_index[word] = i
            new_w2i = defaultdict(lambda: self._word_to_index["<UNK>"])
            new_w2i.update(self._word_to_index)
            self._word_to_index = new_w2i
            for key, value in self._word_to_index.items():
                self._index_to_word[value] = key
        self.add_word("<BOS>")
        logger.info(f"VOCAB_SIZE {self.vocab_size()}")

    def finalize(self):
        """
        This is used for debug
        """
        self._word_to_index = dict(
            self._word_to_index
        )  # To avoid using mis-typing tokens

    def add_word(self, word):
        assert word not in self._word_to_index
        self._word_to_index[word] = self.vocab_size()  # vocab_size() is the
        self._index_to_word[self.vocab_size()] = word

    def split_sentence(self, sentence):
        """ Break sentence into a list of words and punctuation """
        toks = []
        for word in [
            s.strip().lower()
            for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip())
            if len(s.strip()) > 0
        ]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(
                c in "." for c in word
            ):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def vocab_size(self):
        return len(self._index_to_word)

    def word_to_index(self, word):
        return self._word_to_index[word]

    def encode_sentence(self, sentences, seps=None, get_mask=False):
        assert get_mask is False
        if len(self._word_to_index) == 0:
            sys.exit("Tokenizer has no vocab")
        encoding = []
        if type(sentences) is not list:
            sentences = [sentences]
            seps = [seps]
        for sentence, sep in zip(sentences, seps):
            if sep is not None:
                encoding.append(self._word_to_index[sep])
            for word in self.split_sentence(sentence)[::-1]:  # reverse input sentences
                if word in self._word_to_index:
                    encoding.append(self._word_to_index[word])
                else:
                    encoding.append(self._word_to_index["<UNK>"])
        encoding.append(self._word_to_index["<EOS>"])
        if len(encoding) < self.encoding_length:
            encoding += [self._word_to_index["<PAD>"]] * (
                self.encoding_length - len(encoding)
            )

        # cut off the LHS of the encoding if it's over-size (e.g., words from the end of an individual command,
        # favoring those at the beginning of the command (since inst word order is reversed) (e.g., cut off the early
        # instructions in a dialog if the dialog is over size, preserving the latest QA pairs).
        prefix_cut = max(0, len(encoding) - self.encoding_length)
        return np.array(encoding[prefix_cut:])

    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self._word_to_index["<PAD>"]:
                break
            else:
                try:
                    word = self._index_to_word[ix]
                    sentence.append(word)
                except:
                    pass
                    # print("Missing index %d" % ix )
        return " ".join(sentence[::-1])  # unreverse before output

    def shrink(self, inst):
        """
        :param inst:    The id inst
        :return:  Remove the potential <BOS> and <EOS>
                  If no <EOS> return empty list
        """
        if len(inst) == 0:
            return inst
        end = np.argmax(
            np.array(inst) == self._word_to_index["<EOS>"]
        )  # If no <EOS>, return empty string
        if len(inst) > 1 and inst[0] == self._word_to_index["<BOS>"]:
            start = 1
        else:
            start = 0
        # print(inst, start, end)
        return inst[start:end]


def dialog_to_string(dialog):
    # dia_inst = ""
    sentences = []
    seps = []
    for turn in dialog:
        sentences.append(turn["message"])
    return " ".join(sentences)


def angle_feature(heading, elevation):
    import math

    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    return np.array(
        [
            math.sin(heading),
            math.cos(heading),
            math.sin(elevation),
            math.cos(elevation),
        ],
        dtype=np.float32,
    )


def get_point_angle_feature(baseViewId=0):
    sim = new_simulator()

    angle_feat_size = 4

    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            sim.newEpisode(
                ["ZMojNkEp431"],
                ["2f4d90acd4024c269fb0efe49a8ac540"],
                [0],
                [math.radians(-30)],
            )
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading

        feature[ix, :] = angle_feature(heading, state.elevation)
    return feature


def get_all_point_angle_feature():
    return [get_point_angle_feature(baseViewId) for baseViewId in range(36)]


def new_simulator():
    import MatterSim

    # Simulator image parameters
    WIDTH = 600
    HEIGHT = 600
    VFOV = 80

    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.initialize()

    return sim


def length2mask(length, device, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (
        torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
        > (torch.LongTensor(length) - 1).unsqueeze(1)
    ).to(device)
    return mask


def copy_dialog_history(obs):
    new_obs = []
    for ob in obs:
        new_obs.append(
            {
                "inst_idx": ob["inst_idx"],
                "scan": ob["scan"],
                "viewpoint": ob["viewpoint"],
                "viewIndex": ob["viewIndex"],
                "heading": ob["heading"],
                "elevation": ob["elevation"],
                "feature": ob["feature"],
                "candidate": ob["candidate"],
                "step": ob["step"],
                "navigableLocations": ob["navigableLocations"],
                "instructions": ob["instructions"],
                "teacher": ob["teacher"],
                "generated_dialog_history": copy.deepcopy(
                    ob["generated_dialog_history"]
                ),
                "instr_encoding": ob["instr_encoding"],
                "nav_instr_encoding": ob["nav_instr_encoding"],
                "ora_instr_encoding": ob["ora_instr_encoding"],
                "distance": ob["distance"],
                "action_probs": ob["action_probs"],
            }
        )
    return new_obs


# Determine next model inputs
def next_decoder_input(
    logit, feedback, temperature=None, all_env_action=[], batch_size=100, target=None
):
    a_t = None
    if "temperature" in feedback or "penalty" in feedback:
        logit = logit * 1.0 / temperature
    if "penalty" in feedback and len(all_env_action) > 0:
        taken_actions = {}
        for turn in all_env_action:
            for i in range(batch_size):
                if i not in taken_actions:
                    taken_actions[i] = set()
                taken_actions[i].add(turn[i])
        for i in range(batch_size):
            for v in taken_actions[i]:
                logit[i, v] *= temperature
    if feedback == "teacher":
        a_t = target  # teacher forcing
    elif feedback == "argmax":
        _, a_t = logit.max(1)  # student forcing - argmax
        a_t = a_t.detach()
    elif feedback == "sample" or feedback == "temperature" or feedback == "penalty":
        probs = F.softmax(logit, dim=1)
        m = D.Categorical(probs)
        a_t = m.sample()  # sampling an action from model
    elif feedback == "topk":
        k = 3
        topk, sorted_indices = torch.topk(logit, k, dim=1)
        probs = F.softmax(topk, dim=1)
        m = D.Categorical(probs)
        s = m.sample()
        a_t = sorted_indices.gather(1, s.unsqueeze(1)).squeeze()
    elif "nucleus" in feedback:
        p = 0.4
        coin = torch.ones(batch_size).float() * p
        b = D.Bernoulli(coin)
        flip = b.sample().int().to(args.device)
        u = D.Uniform(torch.zeros(batch_size), torch.ones(batch_size) * logit.size()[1])
        uniform = u.sample().int().to(args.device)
        probs = F.softmax(logit, dim=1)
        m = D.Categorical(probs)
        categorical = m.sample().int()
        stack = torch.stack([uniform, categorical], 1)
        a_t = stack.gather(1, flip.unsqueeze(1).long()).squeeze()
    else:
        sys.exit("Invalid feedback option")
    return a_t


def get_optimizer_constructor(optim="rms"):
    optimizer = None
    if optim == "rms":
        logger.info("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == "adam":
        logger.info("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == "sgd":
        logger.info("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif optim == "adamax":
        logger.info("Optimizer: adamax")
        optimizer = torch.optim.Adamax
    else:
        assert False
    return optimizer
