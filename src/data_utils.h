#pragma once

#include <sentencepiece_processor.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include "tokenizer_albert.h"
#include "transformer_example.h"

template <typename ExampleType = TransformerExample>
std::vector<ExampleType> readCsvFile(const std::string &filepath);

torch::Tensor _label_to_tensor(const std::string &label,
                               torch::TensorOptions topts);

template <typename TokenizerType = TokenizerBase,
          typename TransformerSingleExample = TransformerExample,
          typename TransformerSingleFeatures = TransformerFeatures<>>
class SST2 : public torch::data::datasets::Dataset<
                 SST2<TokenizerType, TransformerSingleExample,
                      TransformerSingleFeatures>,
                 TransformerSingleFeatures> {
public:
  // Loads the SST dataset from the `filepath` path.
  //
  // The supplied `filepath` path should be a tsv file with the sentence
  // followed by the label.
  explicit SST2(const std::string &fp, const std::string &pretrained_dir,
                const int msl);

  // Returns the `TransformerSingleExample` at the given `index`.
  TransformerSingleFeatures get(size_t index) override;

  // Returns the `torch::Tensor`s for the data and targets of a batch
  // (at::ArrayRef) pair<torch::Tensor, torch::Tensor>
  // get_batch(torch::ArrayRef<size_t> indices) override;

  // Returns the size of the dataset.
  torch::optional<size_t> size() const override;

  // text to token_ids
  virtual void t2id(std::string &s);

  // Returns all examples as a vector.
  const std::vector<TransformerSingleExample> &examples() const;

  // Returns all targets stacked into a single tensor.
  // const torch::Tensor& targets() const;

private:
  std::vector<TransformerSingleExample> examples_;
  TokenizerType tokenizer_;
  int msl_; // maximum sequence length
};

template class SST2<TokenizerAlbert>;
template class SST2<>;
