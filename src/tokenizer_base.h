#pragma once

#include "config_utils.h"
#include "transformer_example.h"

#include <string>
#include <tuple>
#include <vector>

using namespace std;

using tpenc = tuple<vector<long>, vector<long>, vector<long>>;

class TokenizerBase {
public:
  TokenizerBase(string bos_token, string eos_token,
			     string unk_token, string sep_token,
			     string pad_token, string cls_token,
			     string mask_token,
			     vector<string> additional_special_tokens,
			     long _pad_token_type_id = 0);
  TokenizerBase(const char *pretrained_dir);
  TokenizerBase(TransformersTokenizerConfigs configs);
  ~TokenizerBase() {};
  virtual vector<string> tokenize(string &text);
  virtual vector<long> convert_tokens_to_ids(vector<string> &tokens);
  virtual tpenc encode(string &text_a, string &text_b,
                       bool add_special_tokens, size_t max_length,
                       size_t stride,
                       const char *truncation_strategy,
                       bool pad_to_max_length);
  virtual vector<long>
  create_token_type_ids_from_sequences(vector<long> &token_ids_0,
                                       vector<long> &token_ids_1);
  virtual vector<string> convert_ids_to_tokens(vector<long> &ids);
  virtual string convert_tokens_to_string(vector<string> &tokens);
  virtual string decode(vector<long> &token_ids);

private:
  // variables for state of tokenizer
  string bos_token;
  string eos_token;
  string unk_token;
  string sep_token;
  string pad_token;
  string cls_token;
  string mask_token;
  vector<string> additional_special_tokens;
  long _pad_token_type_id;
  // private functions to be overrided
  //virtual vector<string> _tokenize(string text) override;
};
