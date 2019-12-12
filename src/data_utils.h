#pragma once

#include <sentencepiece_processor.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

std::vector<std::pair<std::string, int64_t>> readCsvFile(const std::string& filepath);

//sentencepiece::SentencePieceProcessor loadSpModel(const std::string& modelpath);

class SST2 : public torch::data::datasets::Dataset<SST2> {
 public:
    // The mode in which the dataset is loaded
    //enum Mode { kTrain, kTest };

    // Loads the SST dataset from the `filepath` path.
    //
    // The supplied `filepath` path should be a tsv file with the sentence followed 
    // by the label.  
    //explicit CIFAR10(const std::string& root, Mode mode = Mode::kTrain);
    explicit SST2(const std::string& fp, const std::string& sp);

    // Returns the `Example` at the given `index`.
    torch::data::Example<> get(size_t index) override;

    // Returns the size of the dataset.
    torch::optional<size_t> size() const override;
    
    // text to token_ids
    virtual void t2id(std::string& s, sentencepiece::SentencePieceProcessor& sp);

    // Returns all examples as a vector.
    const std::vector<std::pair<std::string, int64_t>>& examples() const;

    // Returns all targets stacked into a single tensor.
    //const torch::Tensor& targets() const;

 private:
    std::vector<std::pair<std::string, int64_t>> examples_;
    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
    //sentencepiece::SentencePieceProcessor processor_;
};

