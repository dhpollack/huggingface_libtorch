#pragma once

#include "config_utils.h"
#include "transformer_example.h"

#include <string>
#include <tuple>
#include <vector>

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
