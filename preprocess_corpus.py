#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from enlp.twibo import TwiboParser
from enlp.corpus.emoint import AngerTrain, FearTrain, JoyTrain, SadnessTrain

if __name__ == "__main__":

    from enlp.corpus.emoint import (AngerTrain, FearTrain, JoyTrain, SadnessTrain,
                                    AngerValid, FearValid, JoyValid, SadnessValid,
                                    AngerTest, FearTest, JoyTest, SadnessTest)

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

    desc = "Help for process_datasets, a script that annotates a list of given corpora"

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--corpus', "-c",
                        nargs='*',
                        choices=Corpora,
                        help="Names of corpus to use. Allowed values are " + ', '.join(Corpora),
                        metavar='')

    args = parser.parse_args()

    if args.corpus:
        corpus_names = args.corpus
    else:
        corpus_names = Corpora.keys()

    for corpus_name in corpus_names:
        Corpus = Corpora[corpus_name]
        corpus = Corpus()
        print "processing " + corpus.name

        parser = TwiboParser()
        parser.batch_parse(corpus.sentences)
        corpus.freeze()
