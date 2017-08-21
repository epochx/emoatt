#!/usr/bin/env bash

#!/bin/bash
DATAPATH=$1

mkdir -p "$DATAPATH"/data/{word_embeddings,pickle,json,model,corpora}
mkdir -p "$DATAPATH"/data/corpus/emoint/

mv *-ratings-0to1.*.txt "$DATAPATH"/data/corpora/emoint/
mv ./glove.twitter.27B.*.txt "$DATAPATH"/data/word_embeddings/
