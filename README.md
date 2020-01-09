# Libtorch + Huggingface Transformers

[![Build Status](https://travis-ci.org/dhpollack/huggingface_libtorch.svg?branch=master)](https://travis-ci.org/dhpollack/huggingface_libtorch)

## Requirements

For now, I am using python to trace [huggingface transformers](https://github.com/huggingface/transformers) model with jit and then load that traced model into this script.  I have also installed [sentencepiece](https://github.com/google/sentencepiece) from source.  Also I am just downloading the sentencepiece model manually and loading it directly.  There are a few things that I would like to do in the future, but as a start, this is what I did to get this working.

[download trained albert model](https://drive.google.com/open?id=1i0rr-ogZ2MDYPpUMBsg-2PV7zVddivJ0) and unzip it somewhere (I put it in the `models` folder).

```
# get the data
scripts/get_data.sh
# get the requirements if you need them
scripts/get_third_party.sh
# after downloading the trained model above and putting it in `models/`
python scripts/trace_albert.py
```

## Install

You can either run `scripts/get_third_party.sh` to install the required libraries or edit the compile.env to point to local versions of the libraries.  I was not able to test using the downloaded copy of boost, but in theory it should work.

```
source compile.env
mkdir build && cd build
cmake ..
make
```

## run
currently, gets 89.2325% accuracy on the dev set vs. 89.2201% last reported by the trained model.

```
./huggingface-albert
```

## too lazy to install...

[colab notebook with GPU](https://colab.research.google.com/drive/1TFZbXhiGBtcWVH3ir9Hb1gLGcJyxzTNS)

You should be able to run this repo from colab with the above link.  I also downloaded the CUDA version of libtorch there.

