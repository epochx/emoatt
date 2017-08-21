#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np
import re
from collections import defaultdict
import random
import warnings
import os

from enlp.corpus.emoint import (AngerTrain, FearTrain, JoyTrain, SadnessTrain,
                                AngerValid, FearValid, JoyValid, SadnessValid,
                                AngerTest, FearTest, JoyTest, SadnessTest)

from enlp.embeddings import (GloveTwitter25, GloveTwitter50,
                             GloveTwitter100, GloveTwitter200)


# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_UNK = "_UNK"


def elongated(token):
    regex = re.compile(r"(.)\1{2}")
    return bool(regex.search(token))

def START_VOCAB(add_padding=False):
    if add_padding:
        return [_PAD, _UNK]
    return [_UNK]


def process_token(token):
    """Pre-processing for each token.

    As Liu et al. 2015 we lowercase all words
    and replace numbers with 'DIGIT'.

    Args:
        token (Token): Token

    Returns:
        str: processed token
    """
    ptoken = token.string.lower()
    if re.match("\\d+", ptoken):
        ptoken = "".join("DIGIT" for _ in ptoken)
    return ptoken



def sent_to_bin_feats(sentence, funcs):
    """Generates binary one-hot features from a sentence.
    Applies each function in funcs to each token
    in the provided sentence.

    Args:
        sentence (Sentence):
            Sentence object
        funcs (list):
            list of functions to apply to
            each token in the sentence

    Returns:
        numpy.array : dim=(len(sentence), len(feat_funcs)
    """
    if not sentence.is_tokenized:
        raise Exception("Sentence not tokenized")
    matrix = np.zeros((len(sentence), len(funcs)))
    for i, token in enumerate(sentence):
        for j, func in enumerate(funcs):
            if func(token):
                matrix[i, j] = 1
    return matrix.tolist()


def build_token_vocab(sentences, process_token_func, min_freq=1, add_padding=True):
    """
    TODO
    """
    counts = defaultdict(int)
    sentence_seqs = []

    for sentence in sentences:
        sentence_seq = []
        for word in sentence.tokens:
            processed_token = process_token_func(word)
            counts[processed_token] += 1
            sentence_seq.append(processed_token)
        sentence_seqs.append(sentence_seq)

    # get words with low counts but not labeled as aspects
    highcounts = {word for word, count in counts.iteritems() if count > min_freq}

    word2idx = {token: i for i, token in enumerate(START_VOCAB(add_padding) + list(highcounts))}

    return word2idx

def build_intensity_values(sentences):
    return [s.intensity for s in sentences]

def build_class_vocab(sentences):
    labels =  set([s.sentiment for s in sentences])
    class2idx = {label: i for i, label in enumerate(labels)}
    return class2idx


def build_sequences(sentences, token2idx, process_token_func):
    sentence_seqs = []
    for sentence in sentences:
        sentence_seq = []
        for token in sentence.tokens:
            processed_token = process_token_func(token)
            sentence_seq.append(token2idx.get(processed_token, token2idx[_UNK]))
        sentence_seqs.append(sentence_seq)
    return sentence_seqs


def build_sequence_classes(sentences, class2idx):
    return [class2idx[s.sentiment] for s in sentences]


def build_sequence_ids(sentences):
    return [s.id for s in sentences]


def split_list(examples, dev_ratio=0.9, test_ratio=None):
    """
    :param examples: list
    :param dev_ratio: 0.9
    :param test_ratio: to split train/test
    :param generate_test: (otherwise valid=test)
    :return:
    """
    if test_ratio:
        train, valid, test = [], [], []
        dev_size = int(len(examples)*test_ratio)
        train_size = int(dev_size * dev_ratio)
        for example in examples:
            r = random.random()
            if r <= test_ratio and len(train) + len(valid) < dev_size:
                rr = random.random()
                if rr < dev_ratio and len(train) < train_size:
                    train.append(example)
                else:
                    valid.append(example)
            else:
                test.append(example)
        return train, valid, test
    else:
        train, valid = [], []
        train_size = int(len(examples)*dev_ratio)
        for example in examples:
            r = random.random()
            if r <= dev_ratio and len(train) < train_size:
                train.append(example)
            else:
                valid.append(example)
        return train, valid


def kfolds_split_list(dataset, folds):
    """

    :param dataset:
    :param folds:
    :return: List of tuples.
    """
    result = []
    foldsize = 1.0 * len(dataset) / folds
    rest = 1.0 * len(dataset) % folds
    for f in range(1, folds + 1):
        start = (f - 1) * foldsize
        end = f * foldsize
        train_size = foldsize * (folds - 1)
        if f == folds:
            end += rest
        train, valid, test = [], [], []
        for i, example in enumerate(dataset):
            # test
            if start <= i < end:
                test.append(example)
            # development
            else:
                rr = random.random()
                if rr < 0.9 and len(train) < train_size:
                    train.append(example)
                else:
                    valid.append(example)
        result.append((train, valid, test))
    return result


def build_json_dataset(json_path, train_corpora, valid_corpora=None, test_corpora=None, min_freq=1, feat_funcs=(),
                       add_padding=True, embeddings=None, test_ratio=0.8, folds=1):
    """
    Generates JSON file/s for training a model on the provided corpus. If using folds,
    it generates a folder with the one JSON per fold.

    Args:
        json_path (str):
            Json path
        train_corpus (Corpus):
            Corpus object
        test_corpus (Corpus):
            Corpus object
        min_freq (int)
            Min frequency to add the token to the vocabulary.
        feat_funcs (list):
            list of functions to extract binary features from tokens, None if no
            binary features should be extracted (default=None)
        add_padding (bool)
            True to add the padding to the vocabulary.
        embeddings (Embeddings):
            Embeddings object
        test_ratio (float):
            Development/Test ratio when test set not given (default=0.8)
        folds (int)
            Use folds
        sentiment (bool):
            True if add sentiment (-1, 0 or 1) label to the aspect
        joint (bool)
            True if separate

    Returns:
        name (str) of the processed corpus if successful,
        or None if it fails
    """
    only = True

    all_train = []
    for train_corpus in train_corpora:
        all_train += train_corpus.sentences

    if folds == 1:

        if not valid_corpora and not test_corpora:
            partitions = [split_list(all_train, test_ratio=test_ratio)]
        else:
            if test_corpora:
                all_test = []
                for test_corpus in test_corpora:
                    all_test += test_corpus.sentences
                if valid_corpora:
                    all_valid = []
                    for valid_corpus in valid_corpora:
                        all_valid += valid_corpus.sentences
                else:
                    all_train, all_valid = split_list(all_train)

            else:
                if valid_corpora:
                    _, ___, all_test = split_list(all_train, test_ratio=test_ratio)
                    all_valid = []
                    for valid_corpus in valid_corpora:
                        all_valid += valid_corpus.sentences
                else:
                    all_train, all_valid, all_test = split_list(all_train,
                                                                test_ratio=test_ratio)
            partitions = [(all_train, all_valid, all_test)]
    else:
        partitions = kfolds_split_list(all_train, folds=folds)
        only = False

    for f, partition in enumerate(partitions):

        if "TwiboParser" in train_corpus.pipeline:
            pipeline = "TwiboParser"
        else:
            raise NotImplementedError("Pipeline not supported")

        jsondic = dict()
        train_sentences, valid_sentences, test_sentences = partition

        token2idx = build_token_vocab(train_sentences, process_token, min_freq=min_freq)

        train_x = build_sequences(train_sentences, token2idx, process_token)
        train_y = build_intensity_values(train_sentences)
        train_ids = build_sequence_ids(train_sentences)

        valid_x = build_sequences(valid_sentences, token2idx, process_token)
        valid_y = build_intensity_values(valid_sentences)
        valid_ids = build_sequence_ids(valid_sentences)

        test_x = build_sequences(test_sentences, token2idx, process_token)
        test_y = build_intensity_values(test_sentences)
        test_ids = build_sequence_ids(test_sentences)

        class2idx = build_class_vocab(train_sentences)
        classes = False
        if len(class2idx) > 1:
            classes = True
            train_z = build_sequence_classes(train_sentences, class2idx)
            valid_z = build_sequence_classes(valid_sentences, class2idx)
            test_z = build_sequence_classes(test_sentences, class2idx)

        if feat_funcs:
            train_feat_x = [sent_to_bin_feats(s, feat_funcs) for s in train_sentences]
            valid_feat_x = [sent_to_bin_feats(s, feat_funcs) for s in valid_sentences]
            test_feat_x = [sent_to_bin_feats(s, feat_funcs) for s in test_sentences]

        if embeddings:
            vocab_size = len(token2idx)
            vector_size = embeddings.vector_size
            matrix = np.zeros((vocab_size, vector_size), dtype=np.float32)

            # set the the unseen/unknown token embedding
            matrix[token2idx[_UNK]] = embeddings.unseen()

            # set the the padding embedding
            if add_padding:
                matrix[token2idx[_PAD]] = np.random.random((vector_size,))

            for token, idx in token2idx.items():
                if token in embeddings:
                    matrix[idx] = embeddings[token]
                else:
                    matrix[idx] = embeddings.unseen()

            jsondic["embeddings"] = matrix.tolist()
            embeddings_name = embeddings.name
        else:
            embeddings_name = "RandomEmbeddings"

        jsondic["train_x"], jsondic["train_y"], jsondic["train_ids"] = train_x, train_y, train_ids
        jsondic["valid_x"], jsondic["valid_y"], jsondic["valid_ids"] = valid_x, valid_y, valid_ids
        jsondic["test_x"], jsondic["test_y"], jsondic["test_ids"] = test_x, test_y, test_ids
        jsondic["token2idx"] = token2idx

        if feat_funcs:
            jsondic["featdim"] = len(feat_funcs)
            jsondic["train_feat_x"] = train_feat_x
            jsondic["valid_feat_x"] = valid_feat_x
            jsondic["test_feat_x"] = test_feat_x

        jsondic["class2idx"] = class2idx
        jsondic["classdim"] = len(class2idx.keys())

        if classes:
            jsondic["train_z"] = train_z
            jsondic["valid_z"] = valid_z
            jsondic["test_z"] = test_z

        if only:
            if not os.path.isdir(json_path):
                os.makedirs(json_path)

            json_filename = pipeline + "." + "".join([train_corpus.name for train_corpus in train_corpora])

            if valid_corpora:
                json_filename += "".join([valid_corpus.name for valid_corpus in valid_corpora])

            if test_corpora:
                json_filename += "".join([test_corpus.name for test_corpus in test_corpora])
            json_filename += "." + embeddings_name
            json_filename += ".json"

            with open(os.path.join(json_path, json_filename), "wb") as json_file:
                try:
                    json.dump(jsondic, json_file)
                except Exception as e:
                    warnings.warn(str(e))
                    return None
                finally:
                    print("Written " + json_filename)
        else:
            jsondic["fold"] = f
            json_path_name = pipeline + "." \
                             + "".join([train_corpus.name for train_corpus in train_corpora]) + "." \
                             + embeddings_name

            current_json_path = os.path.join(json_path, json_path_name)
            if not os.path.isdir(current_json_path):
                os.makedirs(current_json_path)

            json_filename = str(f) + ".json"
            with open(os.path.join(current_json_path, json_filename), "wb") as json_file:
                try:
                    json.dump(jsondic, json_file)
                except Exception as e:
                    warnings.warn(str(e) + " in " + train_corpus.name)
                finally:
                    print("Written " + json_filename)
if __name__ == "__main__":

    Corpora = {"AngerTrain": AngerTrain,
               "FearTrain": FearTrain,
               "JoyTrain": JoyTrain,
               "SadnessTrain": SadnessTrain,
               "AngerValid": AngerValid,
               "FearValid": FearValid,
               "JoyValid": JoyValid,
               "SadnessValid": SadnessValid,
               "AngerTest": AngerTest,
               "FearTest": FearTest,
               "JoyTest": JoyTest,
               "SadnessTest": SadnessTest}

    desc = "Help for build_json_fold_datasets, a script that takes processed corpora and " \
           "embeddings and generates JSON files for training attenttion-RNNs for aspect-based " \
           "opinion mining using k-fold cross validation"

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("json_path",
                        help="Absolute path to store JSON files" )

    parser.add_argument("--padding", "-p",
                        action='store_true',
                        default=True,
                        help="Add padding. Default=True")

    parser.add_argument("--minfreq", "-mf",
                        type=int,
                        default=1,
                        help="Minimum frequency (default=1)")

    parser.add_argument('--train', "-tr",
                        nargs='*',
                        choices=Corpora,
                        help="Corpus to use as training. Allowed values are " + ', '.join(Corpora),
                        metavar='')

    parser.add_argument('--valid', "-va",
                        nargs='*',
                        choices=Corpora,
                        help="Corpora to use as test. Allowed values are " + ', '.join(Corpora),
                        metavar='')

    parser.add_argument('--test', "-ts",
                        nargs='*',
                        choices=Corpora,
                        help="Corpora to use as test. Allowed values are " + ', '.join(Corpora),
                        metavar='')

    embeddings_dict = {"GloveTwitter25": GloveTwitter25,
                       "GloveTwitter50": GloveTwitter50,
                       "GloveTwitter100": GloveTwitter100,
                       "GloveTwitter200": GloveTwitter200}


    parser.add_argument('--embeddings', "-e",
                        nargs='*',
                        choices=embeddings_dict,
                        help="Names of embeddings to use. Allowed values are " + ', '.join(embeddings_dict),
                        metavar='')

    parser.add_argument('--folds', "-f",
                        type=int,
                        default=1,
                        help="Number of folds to use. Default, no folds (1)")

    parser.add_argument('--ratio', "-r",
                        type=float,
                        default=0.8,
                        help="Train/Test ratio when test set not given. Default=0.8")

    feat_funcs = [lambda t: t.pos == "A",
                  lambda t: t.pos == "!",
                  lambda t: t.pos == "#",
                  lambda t: t.pos == "E",
                  lambda t: t.pos == "@",
                  lambda t: t.pos == "V",
                  lambda t: t.pos == "$",
                  lambda t: t.pos == "O",
                  lambda t: elongated(t.string)]
                  # lambda t: t.iob == "B-NP",
                  # lambda t: t.iob == "B-PP",
                  # lambda t: t.iob == "B-VP",
                  # lambda t: t.iob == "B-ADJP",
                  # lambda t: t.iob == "B-ADVP",
                  # lambda t: t.iob == "I-NP",
                  # lambda t: t.iob == "I-PP",
                  # lambda t: t.iob == "I-VP",
                  # lambda t: t.iob == "I-ADJP",
                  # lambda t: t.iob == "I-ADVP"]
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print("Creating " + args.json_path)
        os.makedirs(args.json_path)

    if args.embeddings:
        embeddings_list = []
        for item in args.embeddings:
            embeddings_list.append(embeddings_dict[item])
    else:
        embeddings_list = [None]

    for Embeddings in embeddings_list:
        if Embeddings:
            print("loading " + str(Embeddings.__name__))
            embeddings = Embeddings()
        else:
            embeddings = Embeddings

        train_corpora= [Corpora[name].unfreeze(["TwiboParser"])
                        for name in args.train]

        if args.valid:
            valid_corpora = [Corpora[name].unfreeze(["TwiboParser"])
                             for name in args.valid]
        else:
            valid_corpora = []

        if args.test:
            test_corpora = [Corpora[name].unfreeze(["TwiboParser"])
                            for name in args.test]
        else:
            test_corpora = []

        build_json_dataset(args.json_path, train_corpora, test_corpora=test_corpora,
                           valid_corpora=valid_corpora, min_freq=args.minfreq,
                           add_padding=args.padding, feat_funcs=feat_funcs,
                           embeddings=embeddings,test_ratio = args.ratio, folds=args.folds)
