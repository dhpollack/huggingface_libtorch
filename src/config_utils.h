#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>

using namespace std;
using json = nlohmann::json;

struct TransformersTokenizerConfig {
  bool do_lower_case;
  vector<string> init_inputs;
  size_t max_len;
};

struct TransformersSpecialTokensMap {
  string cls_token;
  string mask_token;
  string pad_token;
  string sep_token;
  string unk_token;
  string bos_token;  // in sentencepiece models
  string eos_token;  // in sentencepiece models
};

struct TransformersAddedTokens {
  vector<string> added_tokens;
};

struct TransformersTokenizerConfigs {
  TransformersTokenizerConfig tokenizer_config;
  TransformersSpecialTokensMap special_tokens_map;
  TransformersAddedTokens added_tokens;
};

TransformersTokenizerConfigs read_transformers_pretrained(const char *dirpath);
TransformersTokenizerConfig read_transformers_tokenizer_config(ifstream &fd);
TransformersSpecialTokensMap read_transformers_special_tokens_map(ifstream &fd);
TransformersAddedTokens read_transformers_added_tokens(ifstream &fd);
