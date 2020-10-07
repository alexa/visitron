# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import random
import math
import re
import os
import sys
import string
import torch
import numpy as np
from collections import Counter

from utils_data import load_datasets


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


# padding, unknown word, end of sentence
base_vocab = ["<PAD>", "<UNK>", "<EOS>", "<NAV>", "<ORA>", "<TAR>"]
padding_idx = base_vocab.index("<PAD>")


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
        self.word_to_index = {}
        if vocab:
            for i, word in enumerate(vocab):
                self.word_to_index[word] = i

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

    def encode_sentence(self, sentences, seps=None):
        if len(self.word_to_index) == 0:
            sys.exit("Tokenizer has no vocab")
        encoding = []
        if type(sentences) is not list:
            sentences = [sentences]
            seps = [seps]
        for sentence, sep in zip(sentences, seps):
            if sep is not None:
                encoding.append(self.word_to_index[sep])
            for word in self.split_sentence(sentence)[::-1]:  # reverse input sentences
                if word in self.word_to_index:
                    encoding.append(self.word_to_index[word])
                else:
                    encoding.append(self.word_to_index["<UNK>"])
        encoding.append(self.word_to_index["<EOS>"])
        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index["<PAD>"]] * (
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
            if ix == self.word_to_index["<PAD>"]:
                break
            else:
                sentence.append(self.vocab[ix])
        return " ".join(sentence[::-1])  # unreverse before output


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
