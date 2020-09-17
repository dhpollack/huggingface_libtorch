import typer
from typer import Argument

from collections import namedtuple
from enum import Enum
from pathlib import Path
import json
import logging
import numpy as np
import regex
import unicodedata

from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor
from transformers import AutoTokenizer, squad_convert_examples_to_features


logger = logging.getLogger(__name__)

class SquadVersion(str, Enum):
    one = "1"
    two = "2"

GoogleSquadExample = namedtuple("SquadExample", ["qas_id", "question_text", "paragraph_text", "orig_answer_text", "start_position", "is_impossible"])

class GoogleExtractFeatures:
    def __init__(self):
        pass

    @staticmethod
    def read_examples(input_file, is_training=True):
        """ Read Squad Examples

        https://github.com/google-research/albert/blob/master/squad_utils.py#L136

        """
        examples = []
        input_data = json.load(open(input_file))
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    orig_answer_text = None
                    is_impossible = qa.get("is_impossible", False)
                    if is_training:
                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            logger.info("Original raises error because this is impossible, but we are just skipping it...")
                            continue
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            start_position = answer["answer_start"]
                        else:
                            start_position = -1
                            orig_answer_text = ""

                    example = GoogleSquadExample(qas_id, question_text, paragraph_text, orig_answer_text, start_position, is_impossible)
                    examples.append(example)
        return examples

    @staticmethod
    def preprocess_text(inputs, remove_space=True, lower=False):
        if remove_space:
            inputs = " ".join(inputs.strip().split())
        inputs = unicodedata.normalize("NFKD", inputs)
        inputs = "".join(c for c in inputs if not unicodedata.combining(c))
        if lower:
            inputs = inputs.lower()
        return inputs


    def convert_examples_to_features(self, examples, tokenizer, max_seq_length, doc_stride, max_query_length, do_lower_case=False):
        features = []
        for example_index, example in examples:
            # this creates IDs
            query_tokens = tokenizer.encode_ids(self.preprocess_text(example.question_text, lower=do_lower_case))
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[:max_query_length]
            paragraph_text = example.paragraph_text
            # this create BPE-split strings
            para_tokens = tokenizer.tokenize(self.preprocess_text(example.paragraph_text, lower=do_lower_case))
            chartok_to_tok_index = []
            tok_start_to_chartok_index = []
            char_cnt = 0
            for i, token in enumerate(para_tokens):
                # NOTE: Check to make sure huggingface uses this same token
                token = token.replace("▁", " ")
                chartok_to_tok_index.extend([i] * len(token))
                tok_start_to_chartok_index.append(char_cnt)
                char_cnt += len(token)
                tok_end_to_chartok_index(char_cnt - 1)

            tok_cat_text = "".join(para_tokens).replace("▁", " ")
            # original code has a bunch of stuff related to creating a matrix for batching in tensorflow here






app = typer.Typer()

@app.command()
def huggingface_squad_convert(
    squad_version: SquadVersion,
    model_name: str = Argument(..., help="e.g. 'albert-base-v1'"),
    output_dir: Path = Argument(..., help="e.g. 'data/SQuAD/dev1'"),
    data_dir: Path = Path("data/SQuAD")
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if squad_version == SquadVersion.two:
        dev_fn = "dev-v2.0.json"
        processor_cls = SquadV2Processor
    elif squad_version == SquadVersion.one:
        dev_fn = "dev-v1.1.json"
        processor_cls = SquadV1Processor

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    processor = processor_cls()

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
        np.savetxt(output_dir / f"{filename_base}.csv", dataset.tensors[i].cpu().numpy().astype(int), fmt='%i', delimiter=",")

    with open(output_dir / "strings.txt", "w") as fio:
        for t in dataset.tensors[0]:
            fio.write(f"{tokenizer.decode(t)}\n")

if __name__ == "__main__":
    app()
