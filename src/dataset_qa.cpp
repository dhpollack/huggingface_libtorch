#include "dataset_qa.h"

using namespace std;

namespace hflt {

// Constructor
template <typename TokenizerType, typename ExampleType, typename FeaturesType>
TransformerQADS<TokenizerType, ExampleType, FeaturesType>::TransformerQADS(
    const string &pretrained_dir, long maximum_sequence_len,
    const function<vector<ExampleType>(const string &arg)> read_examples_fn,
    const string &read_examples_arg)
    : tokenizer_(pretrained_dir.c_str()),
      examples_(read_examples_fn(read_examples_arg)),
      msl_(maximum_sequence_len) {
  // constructor post-initialization
  vector<pair<size_t, size_t>> doc_span_mapping =
      add_tokens_to_examples(examples_, tokenizer_, msl_);
  items_ = doc_span_mapping;
}

// get()
template <typename TokenizerType, typename ExampleType, typename FeaturesType>
FeaturesType
TransformerQADS<TokenizerType, ExampleType, FeaturesType>::get(size_t index) {
  pair<size_t, size_t> indices = items_[index];
  ExampleType ex = examples_[indices.first];
  pair<size_t, size_t> p_span = ex.p_spans[indices.second];
  FeaturesType features = example_to_features(ex, p_span);
  return features;
}

// size()
template <typename TokenizerType, typename ExampleType, typename FeaturesType>
torch::optional<size_t>
TransformerQADS<TokenizerType, ExampleType, FeaturesType>::size() const {
  torch::optional<size_t> sz(items_.size());
  return sz;
}

// examples()
template <typename TokenizerType, typename ExampleType, typename FeaturesType>
const vector<ExampleType> &
TransformerQADS<TokenizerType, ExampleType, FeaturesType>::examples() const {
  return examples_;
}

// _label_to_tensor()
template <typename TokenizerType, typename ExampleType, typename FeaturesType>
torch::Tensor
TransformerQADS<TokenizerType, ExampleType, FeaturesType>::_label_to_tensor(
    const string &label, torch::TensorOptions &topts) {
  vector<long> lv;
  stringstream ss(label);
  std::transform(istream_iterator<long>(ss), istream_iterator<long>(),
                 std::back_inserter(lv), [](long x) { return x; });
  return torch::from_blob(lv.data(), {(long)lv.size()}, topts).clone();
}

// example_to_features()
template <typename TokenizerType, typename ExampleType, typename FeaturesType>
FeaturesType
TransformerQADS<TokenizerType, ExampleType, FeaturesType>::example_to_features(
    ExampleType &example, pair<size_t, size_t> &p_span) {
  // google:
  // https://github.com/google-research/bert/blob/master/extract_features.py
  // huggingface:
  // https://github.com/huggingface/transformers/blob/master/src/transformers/data/processors/squad.py#L91
  // The Google version just converts examples to features, while the
  // huggingface version creates a feature for each span
  auto opts_data = torch::TensorOptions().dtype(torch::kLong);
  auto max_len = static_cast<size_t>(msl_);
  vector<int> q_tokens(example.q_tokens.begin(), example.q_tokens.end());
  q_tokens.insert(q_tokens.begin(), tokenizer_.cls_token_id());
  q_tokens.emplace_back(tokenizer_.sep_token_id());
  vector<int> p_tokens_span(example.p_tokens.begin() + p_span.first,
                            example.p_tokens.begin() + p_span.second);
  p_tokens_span.emplace_back(tokenizer_.sep_token_id());
  vector<long> tokens;
  tokens.insert(tokens.end(), q_tokens.begin(), q_tokens.end());
  tokens.insert(tokens.end(), p_tokens_span.begin(), p_tokens_span.end());
  tokens.resize(max_len);
  vector<long> attention_mask(q_tokens.size() + p_tokens_span.size(), 1);
  attention_mask.resize(max_len, 0);
  vector<long> token_type_ids(q_tokens.size(), 0);
  token_type_ids.resize(max_len, 1);
  vector<long> a_tokens(example.a_tokens.begin(), example.a_tokens.end());
  a_tokens.resize(max_len, 0);
  FeaturesType features(
      {torch::from_blob(tokens.data(), {msl_}, opts_data).clone(),
       torch::from_blob(attention_mask.data(), {msl_}, opts_data).clone(),
       torch::from_blob(token_type_ids.data(), {msl_}, opts_data).clone(),
       torch::arange(0, msl_, opts_data),
       torch::from_blob(a_tokens.data(), {msl_}, opts_data).clone()});
  return features;
}

template class TransformerQADS<TokenizerAlbert>;
template class TransformerQADS<>;

}; // namespace hflt
