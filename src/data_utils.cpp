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

torch::Tensor _label_to_tensor(const string &label,
                               torch::TensorOptions topts) {
  vector<long> lv;
  stringstream ss(label);
  std::transform(istream_iterator<long>(ss), istream_iterator<long>(),
                 std::back_inserter(lv), [](long x) { return x; });
  return torch::from_blob(lv.data(), {(long)lv.size()}, topts).clone();
}

template <typename ExampleType>
vector<ExampleType> readCsvFile(const string &filepath) {
  // This function assumes the csv file is in the format `<sentence>\t<label>`
  vector<ExampleType> examples;
  string line;
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
    ++tok_iter;
    int64_t label = 0;
    istringstream i_str(*tok_iter);
    if (!(i_str >> label)) {
      // this should exclude the header of the csv
      // cout << "found invalid example: " << line << endl;
      continue;
    }
    ExampleType ex = {std::to_string(i), sentence, "", std::to_string(label)};
    examples.push_back(ex);
    ++i;
  }
  return examples;
}

template <typename TokenizerType, typename ExampleType, typename FeaturesType>
SST2<TokenizerType, ExampleType, FeaturesType>::SST2(
    const string &fp, const string &pretrained_dir, const int msl)
    : examples_(readCsvFile<ExampleType>(fp)), msl_(msl),
      tokenizer_(pretrained_dir.c_str()) {}

template <typename TokenizerType, typename ExampleType, typename FeaturesType>
FeaturesType SST2<TokenizerType, ExampleType, FeaturesType>::get(size_t index) {
  // torch::data::Example<>() is {self.data, self.target}
  auto opts_data = torch::TensorOptions().dtype(torch::kLong);
  ExampleType ex = examples_[index];
  // tokenize and tensorize
  FeaturesType features =
      tokenizer_.encode(ex.text_a, ex.text_b, true, msl_, 0, "", true);
  features.label = _label_to_tensor(ex.label, opts_data);
  return features;
}
// dataset size()
template <typename TokenizerType, typename ExampleType, typename FeaturesType>
torch::optional<size_t>
SST2<TokenizerType, ExampleType, FeaturesType>::size() const {
  torch::optional<size_t> sz(examples_.size());
  return sz;
}
// dataset text to token_ids, currently not used.
template <typename TokenizerType, typename ExampleType, typename FeaturesType>
void SST2<TokenizerType, ExampleType, FeaturesType>::t2id(string &s) {
  /*
  vector<int> tokens;
  tokenizer_.processor_->Encode(s, &tokens);
  for (const int token_id : tokens) {
    cout << token_id << endl;
  }
  */
}
