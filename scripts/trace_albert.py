import os
import torch
import transformers

basedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sst2_trained_model_path = os.path.join(basedir, "models", "sst2_trained")
output_path = os.path.join(sst2_trained_model_path, "traced_albert.pt")
print(basedir)
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
print(dummy_input)
model = transformers.AlbertForSequenceClassification.from_pretrained(sst2_trained_model_path, torchscript=True)
traced_model = torch.jit.trace(model, dummy_input)
torch.jit.save(traced_model, output_path)
