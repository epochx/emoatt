#!/usr/bin/python
# -*- coding: utf-8 -*-


from settings import TWIBO_PATH

import os
import tempfile
from subprocess import call
from .rep import Sentence

special_chars = ['{', '}', '$', '&', '%']


class TwiboParser(object):

    def __init__(self, twibo_parser_path=TWIBO_PATH):
        self.path = twibo_parser_path

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def batch_parse(self, sentences):

        if all([isinstance(sentence, Sentence) for sentence in sentences]):
            strings = [sentence.string for sentence in sentences]
        else:
            strings = sentences

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write("\n".join(strings).encode("utf-8"))
        temp_file.close()

        print temp_file.name

        command = "./run.sh " + temp_file.name

        cwd = os.getcwd()
        os.chdir(self.path)
        if call(command, shell=True) != 0:
            raise Exception("Twibo error")

        os.chdir(cwd)

        result_ffile = temp_file.name + ".predict"

        with open(result_ffile, "r") as f:
            lines = f.readlines()

        i = 0
        sentence = sentences[0]
        rels  = []

        for line in lines:
            if line.startswith('\n'):
                sentence.append_tags(rels=rels)
                sentence.pipeline.append(str(self))
                i += 1
                if i == len(sentences):
                    break
                else:
                    sentence = sentences[i]
                    rels = []
                    continue

            lis = line.rstrip().split('\t')
            if len(lis) == 1:
                lis = line.rstrip().split(' ')
            if len(lis) == 0:
                continue
            elif len(lis) == 8:
                dep_index = int(lis[0]) -1
                form = replace_special(lis[1])
                lemma = lis[2]
                #cpos = lis[3]
                pos = lis[4]
                #feat = lis[5]
                head_index = int(lis[6]) - 1

                if lis[7] == '_':
                    dep_label= ''
                else:
                    dep_label = lis[7]

                sentence.append(string=form,
                                lemma=lemma,
                                pos_tag=pos)

                if head_index != -2:
                    rels.append((head_index, dep_label, dep_index))

            #elif len(lis) == 10:
            #    s.add(Token(lis))
            else:
                raise Exception('Data format is broken!')



def replace_special(s):
    for c in special_chars:
        s = s.replace(c, '\\' + c)
    return s
