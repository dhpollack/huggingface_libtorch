#pragma once

#include <sentencepiece_processor.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include "transformer_example.h"
#include "tokenizer_albert.h"

using namespace std;

vector<InputExample>
readCsvFile(const string &filepath);

class SST2 : public torch::data::datasets::Dataset<SST2> {
public:
  // The mode in which the dataset is loaded
  // enum Mode { kTrain, kTest };

  // Loads the SST dataset from the `filepath` path.
  //
  // The supplied `filepath` path should be a tsv file with the sentence
  // followed by the label.
  explicit SST2(const string &fp, const string &pretrained_dir, const int msl);

  // Returns the `Example` at the given `index`.
  torch::data::Example<> get(size_t index) override;

  // Returns the `torch::Tensor`s for the data and targets of a batch
  // (at::ArrayRef) pair<torch::Tensor, torch::Tensor>
  // get_batch(torch::ArrayRef<size_t> indices) override;

  // Returns the size of the dataset.
  torch::optional<size_t> size() const override;

  // text to token_ids
  virtual void t2id(string &s);

  // Returns all examples as a vector.
  const vector<InputExample> &examples() const;

  // Returns all targets stacked into a single tensor.
  // const torch::Tensor& targets() const;

private:
  vector<InputExample> examples_;
  TokenizerAlbert tokenizer_;
  int msl_; // maximum sequence length
};
