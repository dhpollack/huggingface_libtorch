#!/usr/bin/env bash

# stackoverflow seems to think there are better ways of doing the following
cd $(dirname $0)/..

GLUEDLURI=https://raw.githubusercontent.com/nyu-mll/GLUE-baselines/master/download_glue_data.py

mkdir -p data && cd data
wget $GLUEDLURI
python download_glue_data.py --data_dir . --tasks SST
rm download_glue_data.py
