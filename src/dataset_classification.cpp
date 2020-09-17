#include "dataset_classification.h"

using namespace std;

namespace hflt {

// Constructor
template <typename TokenizerType, typename ExampleType, typename FeaturesType>
TransformerClassificationDS<TokenizerType, ExampleType, FeaturesType>::
    TransformerClassificationDS(
        const string &pretrained_dir, long maximum_sequence_len,
        const function<vector<ExampleType>(const string &arg)> read_examples_fn,
        const string &read_examples_arg)
    : tokenizer_(pretrained_dir.c_str()),
      examples_(read_examples_fn(read_examples_arg)),
      msl_(maximum_sequence_len) {}

// get()
template <typename TokenizerType, typename ExampleType, typename FeaturesType>
FeaturesType
TransformerClassificationDS<TokenizerType, ExampleType, FeaturesType>::get(
    size_t index) {
  auto opts_data = torch::TensorOptions().dtype(torch::kLong);
  ExampleType ex = examples_[index];
  // tokenize and tensorize
  FeaturesType features = tokenizer_.encode(ex.text_a, ex.text_b, true, msl_, 0,
                                            "longest_first", true);
  features.label = _label_to_tensor(ex.label, opts_data);
  return features;
}

// size()
template <typename TokenizerType, typename ExampleType, typename FeaturesType>
torch::optional<size_t>
TransformerClassificationDS<TokenizerType, ExampleType, FeaturesType>::size()
    const {
  torch::optional<size_t> sz(examples_.size());
  return sz;
}

// examples()
template <typename TokenizerType, typename ExampleType, typename FeaturesType>
const vector<ExampleType> &
TransformerClassificationDS<TokenizerType, ExampleType,
                            FeaturesType>::examples() const {
  return examples_;
}

// _label_to_tensor()
template <typename TokenizerType, typename ExampleType, typename FeaturesType>
torch::Tensor
TransformerClassificationDS<TokenizerType, ExampleType, FeaturesType>::
    _label_to_tensor(const string &label, torch::TensorOptions &topts) {
  vector<long> lv;
  stringstream ss(label);
  std::transform(istream_iterator<long>(ss), istream_iterator<long>(),
                 std::back_inserter(lv), [](long x) { return x; });
  return torch::from_blob(lv.data(), {(long)lv.size()}, topts).clone();
}

template class TransformerClassificationDS<TokenizerAlbert>;
template class TransformerClassificationDS<>;

}; // namespace hflt
