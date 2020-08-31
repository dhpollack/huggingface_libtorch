#pragma once

#include <torch/data/datasets/base.h>
#include <torch/types.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include <string>
#include <vector>

#include "tokenizer_albert.h"
#include "tokenizer_base.h"
#include "transformer_example.h"

template <typename TokenizerType = TokenizerBase,
          typename TransformerSingleExample = TransformerExample,
          typename TransformerSingleFeatures = TransformerFeatures<>>
class TransformerClassificationDS
    : public torch::data::datasets::Dataset<
          TransformerClassificationDS<TokenizerType, TransformerSingleExample,
                                      TransformerSingleFeatures>,
          TransformerSingleFeatures> {
public:
  // A base dataset for transformers, which loads the model and tokenizer from
  // `pretrained_dir` and populates the `examples_` member using the function
  // `read_examples` and `read_examples_arg`.
  //
  // TODO: make `read_examples` and `read_examples_arg` more generic with
  // variadic templates
  //
  // The supplied `filepath` path should be a tsv file with the sentence
  // followed by the label.
  explicit TransformerClassificationDS(
      const std::string &pretrained_dir, long maximum_sequence_len,
      const std::function<
          std::vector<TransformerSingleExample>(const std::string &arg)>
          read_examples_fn,
      const std::string &read_examples_arg);

  // Returns the `TransformerSingleExample` at the given `index`.
  virtual TransformerSingleFeatures get(size_t index) override;

  // Returns the size of the dataset.
  torch::optional<size_t> size() const override;

  // Returns all examples as a vector.
  const std::vector<TransformerSingleExample> &examples() const;

  // read all examples
  // virtual std::vector<TransformerSingleExample> read_examples(const
  // std::string &arg) override;

private:
  torch::Tensor _label_to_tensor(const std::string &label,
                                 torch::TensorOptions &topts);
  TokenizerType tokenizer_;
  std::vector<TransformerSingleExample> examples_;
  long msl_; // maximum sequence length
};
