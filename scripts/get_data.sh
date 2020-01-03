#!/usr/bin/env bash

# stackoverflow seems to think there are better ways of doing the following
cd $(dirname $0)/..

SST2URI=http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip

mkdir -p data/SST-2 && cd data/SST-2
wget $SST2URI -O sst2.zip
unzip sst2.zip

