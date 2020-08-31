import os
from pathlib import Path
from pydoc import locate

import torch
import transformers
import typer

app = typer.Typer()


@app.command()
def trace_classification_model(
    model_path: Path = "./models/sst2_trained",
    output_name: str = "traced_albert.pt",
    hf_model_class: str = "transformers.AlbertForSequenceClassification",
    hf_tokenizer_class: str = "transformers.AlbertTokenizer",
):
    output_path = model_path / output_name
    tokenizer_cls = locate(hf_tokenizer_class)
    tokenizer = tokenizer_cls.from_pretrained(str(model_path))
    tokens = tokenizer.encode(
        "this is a test", add_special_tokens=True, return_tensors="pt"
    ).flatten()
    tokens_len = tokens.size(0)
    token_ids = torch.zeros(128, dtype=torch.long)
    token_ids[:tokens_len] = tokens
    token_ids.unsqueeze_(0)
    attention_mask = torch.ones(128, dtype=torch.long)
    attention_mask[:tokens_len] = 0
    attention_mask.unsqueeze_(0)
    token_type_ids = (attention_mask == 0).to(torch.long)
    position_ids = torch.arange(0, 128, dtype=torch.long)
    dummy_input = [token_ids, attention_mask, token_type_ids, position_ids]
    model_cls = locate(hf_model_class)
    model = model_cls.from_pretrained(str(model_path), torchscript=True)
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, str(output_path))


if __name__ == "__main__":
    app()
