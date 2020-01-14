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

std::vector<std::pair<std::string, int64_t>>
readCsvFile(const std::string &filepath) {
  // This function assumes the csv file is in the format `<sentence>\t<label>`
  std::vector<std::string> sentences;
  std::vector<int64_t> labels;
  std::vector<std::pair<std::string, int64_t>> examples;
  std::string line;
  std::pair<std::string, int64_t> p;
  std::ifstream ifs(filepath);
  if (!ifs.is_open()) {
    return examples;
  }
  typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
  boost::char_separator<char> sep("\t");
  while (std::getline(ifs, line)) {
    tokenizer tokens(line, sep);
    tokenizer::iterator tok_iter = tokens.begin();
    std::string sentence = *tok_iter;
    p.first = sentence;
    ++tok_iter;
    int64_t label = 0;
    std::istringstream i_str(*tok_iter);
    if (!(i_str >> label)) {
      // this should exclude the header of the csv
      //std::cout << "found invalid example: " << line << std::endl;
      continue;
    }
    p.second = label;
    examples.push_back(p);
    sentences.push_back(p.first);
    labels.push_back(p.second);
  }
  return examples;
}

SST2::SST2(const std::string &fp, const std::string &sp, const int msl)
    : examples_(readCsvFile(fp)), msl_(msl) {
  std::shared_ptr<sentencepiece::SentencePieceProcessor> tok_model(
      new sentencepiece::SentencePieceProcessor());
  tok_model->LoadOrDie(sp);
  processor_ = std::move(tok_model);
}

// dataset get
torch::data::Example<> SST2::get(size_t index) {
  // torch::data::Example<>() is {self.data, self.target}
  auto opts_data = torch::TensorOptions().dtype(torch::kInt32);
  auto opts_tgt = torch::TensorOptions().dtype(torch::kInt32);
  auto p = examples_[index];
  // tensorize data
  std::vector<int> token_ids_raw;
  processor_->Encode(p.first, &token_ids_raw);
  token_ids_raw.insert(token_ids_raw.begin(), 2);
  // TODO: check length of token_ids_raw then trim if needed then add EOS
  token_ids_raw.push_back(3);
  int64_t data_size = token_ids_raw.size();
  token_ids_raw.resize(msl_, 0);
  torch::Tensor token_ids =
      torch::from_blob(token_ids_raw.data(), {msl_}, opts_data)
          .to(torch::kInt64);
  // attention mask
  std::vector<int> attention_mask_raw(data_size, 1);
  attention_mask_raw.resize(msl_, 0);
  torch::Tensor attention_mask =
      torch::from_blob(attention_mask_raw.data(), {msl_}, opts_data)
          .to(torch::kInt64);
  // token_type_ids
  std::vector<int> token_type_ids_raw(data_size, 0);
  token_type_ids_raw.resize(msl_, 1);
  torch::Tensor token_type_ids =
      torch::from_blob(token_type_ids_raw.data(), {msl_}, opts_data)
          .to(torch::kInt64);
  // position ids
  torch::Tensor position_ids =
      torch::arange(0, msl_, opts_data).to(torch::kInt64);
  // stack data tensors
  // TODO: figure out how to use a custom type instead of torch::data::Example
  torch::Tensor ret_data = torch::stack(
      {token_ids, attention_mask, token_type_ids, position_ids}, 0);
  // tensorize label
  std::vector<int64_t> label_raw{p.second};
  int64_t label_size = label_raw.size();
  torch::Tensor label =
      torch::from_blob(label_raw.data(), {label_size}, opts_tgt)
          .to(torch::kInt64);
  return {ret_data, label};
}
// dataset size()
torch::optional<size_t> SST2::size() const {
  torch::optional<size_t> sz(examples_.size());
  return sz;
}
// dataset text to token_ids, currently not used.
void SST2::t2id(std::string &s) {
  std::vector<int> tokens;
  processor_->Encode(s, &tokens);
  for (const int token_id : tokens) {
    std::cout << token_id << std::endl;
  }
}

