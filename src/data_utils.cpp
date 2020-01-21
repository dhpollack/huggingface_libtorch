#include <boost/tokenizer.hpp>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sentencepiece_processor.h>
#include <sstream>
#include <torch/torch.h>

#include <string>
#include <vector>

#include "data_utils.h"
#include "transformer_example.h"

// 1. load tsv file with boost library
// 2. load sentencepiece model
// 3. tokenize data and check if it's the same as the python data

using namespace std;

vector<InputExample>
readCsvFile(const string &filepath) {
  // This function assumes the csv file is in the format `<sentence>\t<label>`
  //vector<string> sentences;
  //vector<int64_t> labels;
  vector<InputExample> examples;
  string line;
  //pair<string, int64_t> p;
  ifstream ifs(filepath);
  if (!ifs.is_open()) {
    return examples;
  }
  typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
  boost::char_separator<char> sep("\t");
  size_t i = 0;
  while (getline(ifs, line)) {
    tokenizer tokens(line, sep);
    tokenizer::iterator tok_iter = tokens.begin();
    string sentence = *tok_iter;
    //p.first = sentence;
    ++tok_iter;
    int64_t label = 0;
    istringstream i_str(*tok_iter);
    if (!(i_str >> label)) {
      // this should exclude the header of the csv
      //cout << "found invalid example: " << line << endl;
      continue;
    }
    //p.second = label;
    InputExample ex = {std::to_string(i), sentence, "", std::to_string(label)};
    examples.push_back(ex);
    //sentences.push_back(p.first);
    //labels.push_back(p.second);
  }
  return examples;
}

SST2::SST2(const string &fp, const string &pretrained_dir, const int msl)
    : examples_(readCsvFile(fp)), msl_(msl), tokenizer_(pretrained_dir.c_str()) {
}

// dataset get
torch::data::Example<> SST2::get(size_t index) {
  // torch::data::Example<>() is {self.data, self.target}
  auto opts_data = torch::TensorOptions().dtype(torch::kLong);
  auto opts_tgt = torch::TensorOptions().dtype(torch::kInt32);
  auto ex = examples_[index];
  // tokenize text
  auto tokens_tuple = tokenizer_.encode(ex.text_a, ex.text_b, true, msl_, 0, "", true);
  // tensorize data
  torch::Tensor token_ids =
      torch::from_blob(std::get<0>(tokens_tuple).data(), {msl_}, opts_data);
  torch::Tensor attention_mask =
      torch::from_blob(std::get<1>(tokens_tuple).data(), {msl_}, opts_data);
  torch::Tensor token_type_ids =
      torch::from_blob(std::get<2>(tokens_tuple).data(), {msl_}, opts_data);
  torch::Tensor position_ids =
      torch::arange(0, msl_, opts_data);
  // stack data tensors
  // TODO: figure out how to use a custom type instead of torch::data::Example
  torch::Tensor ret_data = torch::stack(
      {token_ids, attention_mask, token_type_ids, position_ids}, 0);
  // tensorize label
  vector<long> label_raw{stol(ex.label)};
  int64_t label_size = label_raw.size();
  torch::Tensor label =
      torch::from_blob(label_raw.data(), {label_size}, opts_tgt).to(torch::kLong);
  return {ret_data, label};
}
// dataset size()
torch::optional<size_t> SST2::size() const {
  torch::optional<size_t> sz(examples_.size());
  return sz;
}
// dataset text to token_ids, currently not used.
void SST2::t2id(string &s) {
  /*
  vector<int> tokens;
  tokenizer_.processor_->Encode(s, &tokens);
  for (const int token_id : tokens) {
    cout << token_id << endl;
  }
  */
}

