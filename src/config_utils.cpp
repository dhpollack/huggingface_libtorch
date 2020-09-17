#include "config_utils.h"

using namespace std;

namespace hflt {

template <typename T> T _contains_or_empty(json j, char const *k) {
  return j.contains(k) ? j[k].get<T>() : T();
}

TransformersTokenizerConfigs read_transformers_pretrained(const char *dirpath) {
  string basedir(dirpath);
  string tokenizer_config_path = basedir + "/tokenizer_config.json";
  ifstream fd_tokenizer_config(tokenizer_config_path);
  TransformersTokenizerConfigs configs;
  TransformersTokenizerConfig tokenizer_config;
  TransformersSpecialTokensMap special_tokens_map;
  TransformersAddedTokens added_tokens;
  if (!fd_tokenizer_config.is_open()) {
    cerr << "something went wrong opening: " << tokenizer_config_path << endl;
  } else {
    tokenizer_config = read_transformers_tokenizer_config(fd_tokenizer_config);
  }
  string special_tokens_map_path = basedir + "/special_tokens_map.json";
  ifstream fd_special_tokens_map(special_tokens_map_path);
  if (!fd_special_tokens_map.is_open()) {
    cerr << "something went wrong opening: " << special_tokens_map_path << endl;
  } else {
    special_tokens_map =
        read_transformers_special_tokens_map(fd_special_tokens_map);
  }
  string added_tokens_path = basedir + "/added_tokens.json";
  ifstream fd_added_tokens(added_tokens_path);
  if (!fd_added_tokens.is_open()) {
    // keep for debugging, but the added tokens json file isn't always created
    // cerr << "something went wrong opening: " << added_tokens_path << endl;
  } else {
    added_tokens = read_transformers_added_tokens(fd_added_tokens);
  }
  // combine into TransformersTokenizerConfigs
  configs = {tokenizer_config, special_tokens_map, added_tokens};
  return configs;
}

TransformersTokenizerConfig read_transformers_tokenizer_config(ifstream &fd) {
  json config;
  fd >> config;
  TransformersTokenizerConfig tc = {
      _contains_or_empty<bool>(config, "do_lower_case"),
      _contains_or_empty<vector<string>>(config, "init_inputs"),
      _contains_or_empty<size_t>(config, "max_len")};
  return tc;
}

TransformersSpecialTokensMap
read_transformers_special_tokens_map(ifstream &fd) {
  json special_tokens;
  fd >> special_tokens;
  TransformersSpecialTokensMap stm = {
      _contains_or_empty<string>(special_tokens, "cls_token"),
      _contains_or_empty<string>(special_tokens, "mask_token"),
      _contains_or_empty<string>(special_tokens, "pad_token"),
      _contains_or_empty<string>(special_tokens, "sep_token"),
      _contains_or_empty<string>(special_tokens, "unk_token"),
      _contains_or_empty<string>(special_tokens, "bos_token"),
      _contains_or_empty<string>(special_tokens, "eos_token")};
  return stm;
}

TransformersAddedTokens read_transformers_added_tokens(ifstream &fd) {
  json added_tokens;
  fd >> added_tokens;
  TransformersAddedTokens at = {added_tokens.empty()
                                    ? vector<string>()
                                    : added_tokens.get<vector<string>>()};
  return at;
}

}; // namespace hflt
