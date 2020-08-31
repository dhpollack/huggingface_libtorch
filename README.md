# Libtorch + Huggingface Transformers

[![Build Status](https://travis-ci.org/dhpollack/huggingface_libtorch.svg?branch=master)](https://travis-ci.org/dhpollack/huggingface_libtorch)

## Requirements

Currently, I have only tested this on Linux (Arch Linux and Ubuntu 18.04 LTS)

To run this repo, you need the following:  

- [x] A modern c++ compiler (newer version of gcc and clang seem to work, although gcc 9 has issues)
- [x] [Libtorch](https://pytorch.org)  
- [x] [Sentencepiece](https://github.com/google/sentencepiece)  
- [x] [Boost](https://boost.org)  
- [x] [nlohmann json](https://github.com/nlohmann/json)  
- [ ] Other Tokenizers (not implemented yet)  

To run the sample, you'll additionally need:  

- [x] [hugginface's transformers](https://github.com/huggingface/transformers)  
- [x] [PyTorch - python version](https://pytorch.org)
- [x] [Anaconda / Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Below are a set of scripts that will download and install the example dataset, a pretrained model, and the various c++ libraries required for this repo.  I am assuming anyone using this has a preexisting install of conda and boost already installed.  If you already have the c++ libraries installed, you can look at `compile.dev` and set the appropriate environmental variables to the locations of your local libraries.  There are some requirements for the above mentioned libraries and you obviously need to install those as well.  You can look at my CI build to see what it would take to run this library from a clean Ubuntu 18.04 LTS system.  

Right now I don't plan on supporting OSX or Windows, although I suspect this could run on OSX if you install all the required libraries manually.  Especially for Windows, I don't have the time nor a Windows system to test this on.  Feel free to make a PR if you want OSX / Mac support.  

Currently, I have only test this with an ALBERT model on a simple classification task (sentiment analysis).  One should be able to use any of the sentencepiece-tokenized models with fairly few modifications and hopefully, I'll be adding the other tokenizers and tasks soon.  

[ALBERT model pretrained on SST-2](https://drive.google.com/open?id=1i0rr-ogZ2MDYPpUMBsg-2PV7zVddivJ0) - download and unzip it (i.e. in a folder called `models`).  

### How to Run Sample / Tests

```sh
# get the data
scripts/get_data.sh
# get the requirements if you need them
scripts/get_third_party.sh
# get model finetuned on the SST-2 dataset
scripts/get_albert_pretrained.sh
# the following sets up a minimal anaconda env to trace a huggingface transformers model
conda create -n hflt python=3.7
conda activate hflt
conda install -c pytorch pytorch cpuonly
pip install transformers typer
# trace the model that we downloaded above
python scripts/trace_model.py
```

## Build from Source

```sh
source compile.env
mkdir build && cd build
cmake ..
make
```

## run
currently, gets 89.2325% accuracy on the dev set vs. 89.2201% last reported by the trained model.

```sh
hflt [model dir] [dataset file]
```

## Build Tests and Run Sample

```sh
source compile.env
mkdir build_debug && cd build_debug
cmake -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Debug ..
make -j $(nproc)
ctest -VV
src/hflt ../models/sst2_trained ../data/SST-2/dev.tsv
```

## too lazy to install...

[colab notebook with GPU](https://colab.research.google.com/drive/1TFZbXhiGBtcWVH3ir9Hb1gLGcJyxzTNS)

You should be able to run this repo from colab with the above link.  I also downloaded the CUDA version of libtorch there.

## training model

I used the transformers library to train the sentiment analysis example with a mostly default parameters.  Here's the command I ran:

```sh
# clone transformers repo for training scripts
git clone https://github.com/huggingface/transformers.git
# single GPU training
python transformers/examples/run_glue.py --task_name sst-2 --data_dir data/SST-2 --model_type albert --model_name_or_path albert-base-v1 --save_steps 5000 --output_dir output --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 32 --overwrite_output_dir
# multi-gpu training
NUM_GPUS=$(nvidia-smi -L | wc -l) python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} transformers/examples/run_glue.py --task_name sst-2 --data_dir data/SST-2 --model_type albert --model_name_or_path albert-base-v1 --save_steps 5000 --output_dir output --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 32 --overwrite_output_dir
```
