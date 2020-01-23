#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>

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
