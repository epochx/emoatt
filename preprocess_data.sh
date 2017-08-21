#!/usr/bin/env bash

DATAPATH=$1

python preprocess_corpus.py --corpus AngerTrain FearTrain JoyTrain SadnessTrain AngerValid FearValid JoyValid SadnessValid AngerTest FearTest JoyTest SadnessTest

python generate_json.py "$DATAPATH" --train AngerTrain --valid AngerValid --test AngerTest --embeddings GloveTwitter50
python generate_json.py "$DATAPATH" --train FearTrain --valid FearValid --test FearTest --embeddings GloveTwitter50
python generate_json.py "$DATAPATH" --train SadnessTrain --valid SadnessValid --test SadnessTest --embeddings GloveTwitter50
python generate_json.py "$DATAPATH" --train JoyTrain --valid JoyValid --test JoyTest --embeddings GloveTwitter50

