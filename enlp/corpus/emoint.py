#!/usr/bin/python
# -*- coding: utf-8 -*-

from os import path
from collections import OrderedDict
from .utils import Corpus, CorpusError
from ..rep import Sentence, Document
from ..settings import CORPORA_PATH

EMOINT_PATH = path.join(CORPORA_PATH, 'emoint')

class EmointCorpus(Corpus):

    filepath = ""

    def __init__(self, filepath=None):
        if filepath:
            self.filepath = filepath
        if self._check():
            self._read()
        else:
            raise CorpusError("Corpus was not properly built. Check for consistency")

    @property
    def sentences(self):
        return self._sentences.values()

    @property
    def documents(self):
        return self._documents.values()

    def _check(self):
        return True

    def _read(self):
        self._counter = 1
        self._sentences = OrderedDict()
        self._documents = OrderedDict()

        # add the single fake review
        document = Document(id=1)
        self._documents[1] = document

        with open(self.filepath, "r") as f:
            for line in f.readlines():
                idx, text, sent, int = line.decode("utf-8").split("\t")
                sentence = Sentence(string=text, id=idx, document=document)
                sentence.sentiment = sent
                int = int.strip()
                if int == "NONE":
                  sentence.intensity = None
                else:
                  sentence.intensity = float(int)
                self._sentences[idx] = sentence

AngerTrain = type("AngerTrain",
                    (EmointCorpus,),
                    {'filepath': path.join(EMOINT_PATH,
                                           'anger-ratings-0to1.train.txt')})

FearTrain= type("FearTrain",
                (EmointCorpus,),
                {'filepath': path.join(EMOINT_PATH,
                                       'fear-ratings-0to1.train.txt')})

JoyTrain = type("JoyTrain",
                (EmointCorpus,),
                {'filepath': path.join(EMOINT_PATH,
                                       'joy-ratings-0to1.train.txt')})

SadnessTrain = type("SadnessTrain",
                    (EmointCorpus,),
                    {'filepath': path.join(EMOINT_PATH,
                                           'sadness-ratings-0to1.train.txt')})

# DEVELOPMENT

AngerValid = type("AngerValid",
                    (EmointCorpus,),
                    {'filepath': path.join(EMOINT_PATH,
                                           'anger-ratings-0to1.dev.gold.txt')})

FearValid= type("FearValid",
                (EmointCorpus,),
                {'filepath': path.join(EMOINT_PATH,
                                       'fear-ratings-0to1.dev.gold.txt')})

JoyValid = type("JoyValid",
                (EmointCorpus,),
                {'filepath': path.join(EMOINT_PATH,
                                       'joy-ratings-0to1.dev.gold.txt')})

SadnessValid = type("SadnessValid",
                    (EmointCorpus,),
                    {'filepath': path.join(EMOINT_PATH,
                                           'sadness-ratings-0to1.dev.gold.txt')})

# TEST

AngerTest = type("AngerTest",
                 (EmointCorpus,),
                 {'filepath': path.join(EMOINT_PATH,
                                           'anger-ratings-0to1.test.target.txt')})

FearTest= type("FearTest",
               (EmointCorpus,),
               {'filepath': path.join(EMOINT_PATH,
                                      'fear-ratings-0to1.test.target.txt')})

JoyTest = type("JoyTest",
               (EmointCorpus,),
               {'filepath': path.join(EMOINT_PATH,
                                      'joy-ratings-0to1.test.target.txt')})

SadnessTest= type("SadnessTest",
                  (EmointCorpus,),
                  {'filepath': path.join(EMOINT_PATH,
                                         'sadness-ratings-0to1.test.target.txt')})
