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
    // set some variables
    int MAXIMUM_SEQUENCE_LENGTH = 128;
    int BATCH_SIZE = 64;
    // read examples
    std::string fp = "/home/david/Programming/experiments/c++/huggingface_albert/data/SST-2/dev-small.tsv";
    auto examples = readCsvFile(fp);
    // load sentencepiece model
    const std::string sp = "/home/david/Programming/experiments/c++/huggingface_albert/models/spiece.model";
    sentencepiece::SentencePieceProcessor processor;
    processor.LoadOrDie(sp);
    auto ds = SST2(fp, sp, MAXIMUM_SEQUENCE_LENGTH)
	.map(torch::data::transforms::Stack<>());; 
    auto dl = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(ds), BATCH_SIZE);
    // load albert model and put into eval mode
    std::string model_path = "/home/david/Programming/experiments/c++/huggingface_albert/models/traced_albert.pt";
    torch::jit::script::Module model;
    try {
	model = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
	std::cerr << "error loading the model\n";
	return -1;
    }
    model.eval();
    // equivalent to with torch.no_grad() in python
    torch::NoGradGuard no_grad_guard;

    // run eval loop
    std::vector<torch::Tensor> preds_vec;
    std::vector<torch::Tensor> labels_vec;
    for (auto& mb : *dl) {
	std::cout << "Batch Size: " << mb.data.size(0) << ", " << mb.data.size(1) << std::endl;
	labels_vec.push_back(mb.target);
	auto token_ids = torch::select(mb.data, 1, 0);
	auto attention_masks = torch::select(mb.data, 1, 1);
	auto token_type_ids = torch::select(mb.data, 1, 2);
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(token_ids);
	inputs.push_back(attention_masks);
	inputs.push_back(token_type_ids);
	auto out = model.forward(inputs).toTuple();
	preds_vec.push_back(out->elements()[0].toTensor());
	std::cout << "Output: " << out->elements()[0].toTensor() << std::endl;
    }
    torch::Tensor preds = torch::cat(preds_vec, 0).argmax(1);
    torch::Tensor labels = torch::cat(labels_vec, 0).flatten();
    float correct = preds.eq(labels).sum().item<float>();
    float total = preds.size(0);
    std::cout << correct / total << std::endl;
    return(1);
}

