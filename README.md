# Libtorch + Huggingface Transformers

## Requirements

For now, I am using python to trace [huggingface transformers](https://github.com/huggingface/transformers) model with jit and then load that traced model into this script.  I have also installed [sentencepiece](https://github.com/google/sentencepiece) from source.  Also I am just downloading the sentencepiece model manually and loading it directly.  There are a few things that I would like to do in the future, but as a start, this is what I did to get this working.

[download trained albert model](https://drive.google.com/open?id=1Hys6cWk6Kdw-4LGnZceto2uK7SIsYKIb) and unzip it somewhere (I put it in the `models` folder).

```
# first install transformers and sentencepiece
mkdir models && cd models
wget https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-spiece.model -O spiece.model
python
# in python repl
import torch, transformers
sst2_trained_model_path = "/path/to/sst2_trained"
tokenizer = transformers.AutoTokenizer.from_pretrained(sst2_trained_model_path)
tokens = tokenizer.encode("this is a test", add_special_tokens=True, return_tensors="pt").flatten()
tokens_len = tokens.size(0)
token_ids = torch.zeros(128, dtype=torch.long)
token_ids[:tokens_len] = tokens
token_ids.unsqueeze_(0)
attention_mask = torch.ones(128, dtype=torch.long)
attention_mask[:tokens_len] = 0
attention_mask.unsqueeze_(0)
token_type_ids = (attention_mask == 0).to(torch.long)
dummy_input = [token_ids, attention_mask, token_type_ids]
model = transformers.AlbertForSequenceClassification.from_pretrained(sst2_trained_model_path, torchscript=True)
traced_model = torch.jit.trace(model, dummy_input)
torch.jit.save(traced_model, "traced_albert.pt")
# exit python repl
```

## Install

You can either run `scripts/get_third_party.sh` to install the required libraries or edit the compile.env to point to local version of the library.  I was not able to test using a local copy of boost, but in theory it should work.

```
source compile.env
mkdir build && cd build
cmake ..
make
```

## run
currently, I am getting about 60% accuracy on the dev set, but I should be getting around 85%.

```
./huggingface-albert
```
