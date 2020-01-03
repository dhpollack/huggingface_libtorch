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
#include "transformer_example.h"

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

SST2::SST2(const std::string& fp, const std::string& sp, const int msl) : examples_(readCsvFile(fp)), msl_(msl) {
    std::shared_ptr<sentencepiece::SentencePieceProcessor> tok_model(new sentencepiece::SentencePieceProcessor());
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
    token_ids_raw.insert(token_ids_raw.begin(), 2);
    // TODO: check length of token_ids_raw then trim if needed then add EOS
    token_ids_raw.push_back(3);
    int64_t data_size = token_ids_raw.size();
    token_ids_raw.resize(msl_, 0);
    torch::Tensor token_ids = torch::from_blob(token_ids_raw.data(), {msl_}, opts_data).to(torch::kInt64);
    // attention mask
    std::vector<int> attention_mask_raw(data_size, 1);
    attention_mask_raw.resize(msl_, 0);
    torch::Tensor attention_mask = torch::from_blob(attention_mask_raw.data(), {msl_}, opts_data).to(torch::kInt64);
    // token_type_ids
    std::vector<int> token_type_ids_raw(data_size, 0);
    token_type_ids_raw.resize(msl_, 1);
    torch::Tensor token_type_ids = torch::from_blob(token_type_ids_raw.data(), {msl_}, opts_data).to(torch::kInt64);
    // position ids
    torch::Tensor position_ids = torch::arange(0, msl_, opts_data).to(torch::kInt64);
    // stack data tensors
    // TODO: figure out how to use a custom type instead of torch::data::Example
    torch::Tensor ret_data = torch::stack({token_ids, attention_mask, token_type_ids, position_ids}, 0); 
    // tensorize label
    std::vector<int64_t> label_raw{p.second};
    int64_t label_size = label_raw.size();
    torch::Tensor label = torch::from_blob(label_raw.data(), {label_size}, opts_tgt).to(torch::kInt64);
    return {ret_data, 
	    label};
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

/* used for testing
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
    //auto item = ds.get(1);
    //std::cout << item.data << std::endl << item.target << std::endl;
    
    auto dl = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(ds),2);
    
    for (auto& mb : *dl) {
	std::cout << "Batch Size: " << mb.data.size(0) << ", " << mb.data.size(1) << std::endl;
    }


    return(1);
}
*/
