#include <boost/tokenizer.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>
#include <sstream>
#include <string>
#include <torch/data/datasets/base.h>
#include <torch/data/transforms.h>
#include <torch/types.h>
#include <tuple>
#include <utility>
#include <vector>
#pragma once

#pragma once

using json = nlohmann::json;

struct TransformersTokenizerConfig {
  bool do_lower_case;
  std::vector<std::string> init_inputs;
  size_t max_len;
};

struct TransformersSpecialTokensMap {
  std::string cls_token;
  std::string mask_token;
  std::string pad_token;
  std::string sep_token;
  std::string unk_token;
  std::string bos_token; // in sentencepiece models
  std::string eos_token; // in sentencepiece models
};

struct TransformersAddedTokens {
  std::vector<std::string> added_tokens;
};

struct TransformersTokenizerConfigs {
  TransformersTokenizerConfig tokenizer_config;
  TransformersSpecialTokensMap special_tokens_map;
  TransformersAddedTokens added_tokens;
};

TransformersTokenizerConfigs read_transformers_pretrained(const char *dirpath);
TransformersTokenizerConfig
read_transformers_tokenizer_config(std::ifstream &fd);
TransformersSpecialTokensMap
read_transformers_special_tokens_map(std::ifstream &fd);
TransformersAddedTokens read_transformers_added_tokens(std::ifstream &fd);
#pragma once

struct TransformerExample {
  std::string guid;
  std::string text_a;
  std::string text_b;
  std::string label;

  TransformerExample() = default;
  TransformerExample(std::string guid, std::string text_a, std::string text_b,
                     std::string label)
      : guid(guid), text_a(text_a), text_b(text_b), label(label){};
};

/// A `TransformerFeatures` from a dataset.
///
/// A dataset consists of data and an associated target (label).
template <typename InputID = torch::Tensor,
          typename AttentionMask = torch::Tensor,
          typename TokenTypeID = torch::Tensor,
          typename PositionID = torch::Tensor, typename Label = torch::Tensor>
struct TransformerFeatures {
  using InputIDType = InputID;
  using AttentionMaskType = AttentionMask;
  using TokenTypeIDType = TokenTypeID;
  using PositionIDType = PositionID;
  using LabelType = Label;

  TransformerFeatures() = default;
  TransformerFeatures(InputID input_ids, AttentionMask attention_mask,
                      TokenTypeID token_type_ids, PositionID position_ids,
                      Label label)
      : input_ids(std::move(input_ids)),
        attention_mask(std::move(attention_mask)),
        token_type_ids(std::move(token_type_ids)),
        position_ids(std::move(position_ids)), label(std::move(label)) {}

  InputIDType input_ids;
  AttentionMaskType attention_mask;
  TokenTypeIDType token_type_ids;
  PositionIDType position_ids;
  LabelType label;
};
#pragma once

class TokenizerBase {
public:
  TokenizerBase(std::string &bos_token, std::string &eos_token,
                std::string &unk_token, std::string &sep_token,
                std::string &pad_token, std::string &cls_token,
                std::string &mask_token,
                std::vector<std::string> &additional_special_tokens,
                long _pad_token_type_id = 0);
  TokenizerBase(const char *pretrained_dir);
  TokenizerBase(TransformersTokenizerConfigs configs);
  ~TokenizerBase(){};
  template <typename T> std::vector<T> tokenize(std::string &text);
  virtual std::vector<long>
  convert_tokens_to_ids(std::vector<std::string> &tokens);
  virtual TransformerFeatures<> encode(std::string &text_a, std::string &text_b,
                                       bool add_special_tokens,
                                       const size_t max_length, size_t stride,
                                       const char *truncation_strategy,
                                       bool pad_to_max_length);
  template <typename T, typename U>
  std::vector<long>
  create_token_type_ids_from_sequences(std::vector<T> &tokens_vec_0,
                                       std::vector<U> &tokens_vec_1);
  virtual std::vector<std::string>
  convert_ids_to_tokens(std::vector<long> &ids);
  virtual std::string
  convert_tokens_to_string(std::vector<std::string> &tokens);
  virtual std::string decode(std::vector<long> &token_ids);
  virtual void truncate_sequences(std::vector<int> &ids_int_a,
                                  std::vector<int> &ids_int_b,
                                  size_t num_tokens_to_remove,
                                  const char *truncation_strategy,
                                  size_t stride);
  virtual int cls_token_id() { return 0; };
  virtual int sep_token_id() { return 0; };
  virtual int pad_token_id() { return 0; };

protected:
  // variables for state of tokenizer
  const std::string bos_token;
  const std::string eos_token;
  const std::string unk_token;
  const std::string sep_token;
  const std::string pad_token;
  const std::string cls_token;
  const std::string mask_token;
  const std::vector<std::string> additional_special_tokens;
  const long _pad_token_type_id;
};
#pragma once

class TokenizerAlbert : public TokenizerBase {
public:
  TokenizerAlbert(const char *pretrained_dir);
  TokenizerAlbert(TransformersTokenizerConfigs configs,
                  const char *spmodel_path);

  TransformerFeatures<> encode(std::string &text_a, std::string &text_b,
                               bool add_special_tokens, size_t max_length,
                               size_t stride, const char *truncation_strategy,
                               bool pad_to_max_length);
  template <typename T> std::vector<T> tokenize(std::string &text);
  std::string decode(std::vector<long> &token_ids);
  int cls_token_id();
  int sep_token_id();
  int pad_token_id();

private:
  std::shared_ptr<sentencepiece::SentencePieceProcessor> processor_;
};
#pragma once

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
#pragma once

using json = nlohmann::json;

struct SquadExample {
  std::string qas_id;
  std::string question_text;
  std::string paragraph_text;
  std::string orig_answer_text;
  int start_position;
  int end_position;
  bool is_impossible;
  std::vector<int> q_tokens;
  std::vector<int> p_tokens;
  std::vector<int> a_tokens;
  std::vector<std::pair<size_t, size_t>> p_spans;
  SquadExample(std::string &qas_id, std::string &question_text,
               std::string &paragraph_text, std::string &orig_answer_text,
               int start_position, int end_position, bool is_impossible)
      : qas_id(qas_id), question_text(question_text),
        paragraph_text(paragraph_text), orig_answer_text(orig_answer_text),
        start_position(start_position), end_position(end_position),
        is_impossible(is_impossible){};
};

std::vector<SquadExample> readSquadExamples(const std::string &input_path);
std::vector<SquadExample> read_squad_examples(std::ifstream &input_file,
                                              bool is_training);

template <typename TokenizerType>
std::vector<std::pair<size_t, size_t>>
add_tokens_to_examples(std::vector<SquadExample> &examples,
                       TokenizerType tokenizer_, long msl_,
                       size_t doc_stride = 128, size_t max_query_length = 64,
                       size_t num_special_tokens = 3);
#pragma once

template <typename TokenizerType = TokenizerBase,
          typename TransformerSingleExample = SquadExample,
          typename TransformerSingleFeatures = TransformerFeatures<>>
class TransformerQADS
    : public torch::data::datasets::Dataset<
          TransformerQADS<TokenizerType, TransformerSingleExample,
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
  explicit TransformerQADS(
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

  TransformerSingleFeatures
  example_to_features(TransformerSingleExample &example,
                      std::pair<size_t, size_t> &p_span);

private:
  torch::Tensor _label_to_tensor(const std::string &label,
                                 torch::TensorOptions &topts);
  TokenizerType tokenizer_;
  std::vector<TransformerSingleExample> examples_;
  std::vector<std::pair<size_t, size_t>> items_;
  long msl_; // maximum sequence length
};
#pragma once

using json = nlohmann::json;

std::vector<TransformerExample>
readGenericJsonFile(const std::string &filepath);
std::vector<TransformerExample> readSST2CsvFile(const std::string &filepath);
#pragma once

template <>
struct torch::data::transforms::Stack<TransformerFeatures<>>
    : public torch::data::transforms::Collation<TransformerFeatures<>> {
  TransformerFeatures<>
  apply_batch(std::vector<TransformerFeatures<>> examples) override {
    std::vector<torch::Tensor> ids, ams, ttis, ps, lbls;
    ids.reserve(examples.size());
    ams.reserve(examples.size());
    ttis.reserve(examples.size());
    ps.reserve(examples.size());
    lbls.reserve(examples.size());
    for (auto &example : examples) {
      ids.emplace_back(std::move(example.input_ids));
      ams.emplace_back(std::move(example.attention_mask));
      ttis.emplace_back(std::move(example.token_type_ids));
      ps.emplace_back(std::move(example.position_ids));
      lbls.emplace_back(std::move(example.label));
    }
    return {torch::stack(ids), torch::stack(ams), torch::stack(ttis),
            torch::stack(ps), torch::stack(lbls)};
  }
};
