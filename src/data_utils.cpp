#include <torch/torch.h>
#include <boost/tokenizer.hpp>
#include <sentencepiece_processor.h>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>

#include <string>
#include <vector>

#include "data_utils.h"

// 1. load tsv file with boost library
// 2. load sentencepiece model
// 3. tokenize data and check if it's the same as the python data

std::vector<std::pair<std::string, int64_t>> readCsvFile(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
	// TODO: throw error or empty result
    }
    std::vector<std::string> sentences;
    std::vector<int64_t> labels;
    std::vector<std::pair<std::string, int64_t>> examples;
    std::string line;
    std::pair<std::string, int64_t> p;
    typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
    boost::char_separator<char> sep("\t");
    while(std::getline(ifs, line)) {
	tokenizer tokens(line, sep);
        tokenizer::iterator tok_iter = tokens.begin(); 
        p.first = *tok_iter;	
	++tok_iter;
	int64_t i = 0;
	std::stringstream i_str(*tok_iter);
	i_str >> i;
	p.second = i;
	examples.push_back(p);
	sentences.push_back(p.first);
	labels.push_back(p.second);
    }
    /* testing code to print out examples
    for (auto x = examples.begin(); x != examples.end(); ++x) {
        std::cout << x->first << " | " << x->second << std::endl;
    }
    */
    return examples;

}

SST2::SST2(const std::string& fp, const std::string& sp) : examples_(readCsvFile(fp)) {
    std::unique_ptr<sentencepiece::SentencePieceProcessor> tok_model(new sentencepiece::SentencePieceProcessor());
    tok_model->LoadOrDie(sp);
    processor_ = std::move(tok_model);
}

// dataset get
torch::data::Example<> SST2::get(size_t index) {
    //torch::data::Example<>() is {self.data, self.target}
    auto opts_data = torch::TensorOptions()
                            .dtype(torch::kInt32);
    auto opts_tgt = torch::TensorOptions()
                           .dtype(torch::kInt32);
    auto p = examples_[index];
    // tensorize data
    std::vector<int> token_ids_raw;
    processor_->Encode(p.first, &token_ids_raw);
    int64_t data_size = token_ids_raw.size();
    torch::Tensor token_ids = torch::from_blob(token_ids_raw.data(), {data_size}, opts_data);
    // tensorize label
    std::vector<int64_t> label_raw{p.second};
    int64_t label_size = label_raw.size();
    torch::Tensor label = torch::from_blob(label_raw.data(), {label_size}, opts_tgt);
    return {token_ids.to(torch::kInt64), label.to(torch::kInt64)};
}
// dataset size()
torch::optional<size_t> SST2::size() const {
    torch::optional<size_t> sz(examples_.size());
    return sz;
}
// dataset text to token_ids, currently not used.
void SST2::t2id(std::string& s) {
    std::vector<int> tokens;
    processor_->Encode(s, &tokens);
    for (const int token_id : tokens) {
	std::cout << token_id << std::endl;
    }
}


int main() {
    std::string fp = "/home/david/Programming/experiments/c++/huggingface_albert/data/SST-2/dev.tsv";
    auto examples = readCsvFile(fp);
    // load sentencepiece model
    const std::string sp = "/home/david/Programming/experiments/c++/huggingface_albert/models/spiece.model";
    sentencepiece::SentencePieceProcessor processor;
    processor.LoadOrDie(sp);
    auto ds = SST2(fp, sp);
    auto item = ds.get(1);
    std::cout << item.data << std::endl << item.target << std::endl;
    /*
    auto dl = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(ds),1);
    for (auto& item : dl) {
	
    }
    */


    return(1);
}
