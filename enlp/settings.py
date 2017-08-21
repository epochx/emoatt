#!/usr/bin/python
# -*- coding: utf-8 -*-

from os import path

CODE_ROOT = path.dirname(path.realpath(__file__))
DATA_PATH = "/path/to/data"
TWIBO_PATH = "/path/to/TweeboParser"

JAVA_HOME = "/usr/lib/jvm/java-8-openjdk-amd64"

CORPORA_PATH = path.join(DATA_PATH, "corpus")
CHUNKLINK_PATH = path.join(CODE_ROOT, "script/mod_chunklink_2-2-2000_for_conll.pl")
PICKLE_PATH = path.join(DATA_PATH, "pickle")
