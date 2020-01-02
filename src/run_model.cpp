#include <torch/script.h>
#include <sentencepiece_processor.h>

#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "data_utils.cpp"

int main() {
    int MAXIMUM_SEQUENCE_LENGTH = 128;
    std::string fp = "/home/david/Programming/experiments/c++/huggingface_albert/data/SST-2/dev.tsv";
    auto examples = readCsvFile(fp);
    // load sentencepiece model
    const std::string sp = "/home/david/Programming/experiments/c++/huggingface_albert/models/spiece.model";
    sentencepiece::SentencePieceProcessor processor;
    processor.LoadOrDie(sp);
    auto ds = SST2(fp, sp, MAXIMUM_SEQUENCE_LENGTH)
	.map(torch::data::transforms::Stack<>());; 
    auto dl = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(ds),2);
    // load albert model
    std::string model_path = "/home/david/Programming/experiments/c++/huggingface_albert/models/traced_albert.pt";
    torch::jit::script::Module model;
    try {
	model = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
	std::cerr << "error loading the model\n";
	return -1;
    }

    for (auto& mb : *dl) {
	std::cout << "Batch Size: " << mb.data.size(0) << ", " << mb.data.size(1) << std::endl;
	auto token_ids = torch::select(mb.data, 1, 0);
	auto attention_masks = torch::select(mb.data, 1, 1);
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(token_ids);
	inputs.push_back(attention_masks);
	auto tuple_type = torch::jit::TupleType::create({torch::jit::IntType::get(), torch::jit::IntType::get()});
	//auto tuple = torch::ivalue::Tuple::create(inputs, tuple_type);
	auto out = model.forward(inputs).toTuple();
	std::cout << "Output: " << out->elements()[0].toTensor() << std::endl;
    }
    return(1);
}

