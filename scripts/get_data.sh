#!/usr/bin/env bash

# stackoverflow seems to think there are better ways of doing the following
cd $(dirname $0)/..

GLUEDLURI=https://raw.githubusercontent.com/nyu-mll/GLUE-baselines/master/download_glue_data.py
SQUADV2BASEURI=https://rajpurkar.github.io/SQuAD-explorer/dataset/

# download GLUE SST-2
mkdir -p data && cd data
wget $GLUEDLURI
python download_glue_data.py --data_dir . --tasks SST
rm download_glue_data.py

# download squad v2
mkdir SQuAD && cd SQuAD
#wget "${SQUADV2BASEURI}/train-v2.0.json"  # ignore trainset for now
wget "${SQUADV2BASEURI}/dev-v2.0.json"

