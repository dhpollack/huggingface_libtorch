import typer

import numpy as np

from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor
from transformers import AutoTokenizer, squad_convert_examples_to_features

output_dir = "data/SQuAD/SQuAD_dev2"
model_name = "albert-base-v1"
data_dir = "data/SQuAD"
dev_fn = "dev-v2.0.json"

tokenizer = AutoTokenizer.from_pretrained(model_name)

processor = SquadV2Processor()

examples = processor.get_train_examples(data_dir, filename=dev_fn)

features, dataset = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=True,
    return_dataset="pt",
    threads=2,
)

for i, filename_base in enumerate(["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]):
    np.savetxt(f"{output_dir}/{filename_base}.csv", dataset.tensors[i].numpy(), delimiter=",")

with open(f"{output_dir}/strings.txt", "w") as fio:
    for t in dataset.tensors[0]:
        fio.write(f"{tokenizer.decode(t)}\n")
