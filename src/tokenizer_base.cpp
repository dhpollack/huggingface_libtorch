#include "tokenizer_base.h"

using namespace std;

TokenizerBase::TokenizerBase(string &bos_token, string &eos_token,
                             string &unk_token, string &sep_token,
                             string &pad_token, string &cls_token,
                             string &mask_token,
                             vector<string> &additional_special_tokens,
                             long _pad_token_type_id)
    : bos_token(bos_token), eos_token(eos_token), unk_token(unk_token),
      sep_token(sep_token), pad_token(pad_token), cls_token(cls_token),
      mask_token(mask_token),
      additional_special_tokens(additional_special_tokens),
      _pad_token_type_id(_pad_token_type_id) {}

TokenizerBase::TokenizerBase(const char *pretrained_dir)
    : TokenizerBase(read_transformers_pretrained(pretrained_dir)) {}

TokenizerBase::TokenizerBase(TransformersTokenizerConfigs configs)
    : TokenizerBase(configs.special_tokens_map.bos_token,
                    configs.special_tokens_map.eos_token,
                    configs.special_tokens_map.unk_token,
                    configs.special_tokens_map.sep_token,
                    configs.special_tokens_map.pad_token,
                    configs.special_tokens_map.cls_token,
                    configs.special_tokens_map.mask_token,
                    configs.added_tokens.added_tokens, 0) {}

template <typename U> vector<U> TokenizerBase::tokenize(string &text) {
  vector<U> tokens;
  return tokens;
}

vector<long> TokenizerBase::convert_tokens_to_ids(vector<string> &tokens) {
  vector<long> ids;
  /*
  for (string &t : tokens) {
    long id = token_to_id_fn(t);
    ids.push_back(id);
  }
  */
  return ids;
}

TransformerFeatures<>
TokenizerBase::encode(string &text_a, string &text_b,
                      bool add_special_tokens = true,
                      const size_t max_length = 512, size_t stride = 0,
                      const char *truncation_strategy = "longest_first",
                      bool pad_to_max_length = false) {
  auto opts_data = torch::TensorOptions().dtype(torch::kLong);
  vector<string> tokens_a = tokenize<string>(text_a);
  vector<long> ids_a = convert_tokens_to_ids(tokens_a);
  vector<string> tokens_b = tokenize<string>(text_b);
  vector<long> ids_b = convert_tokens_to_ids(tokens_b);
  vector<long> ttis = create_token_type_ids_from_sequences(ids_a, ids_b);
  ids_a.insert(ids_a.end(), ids_b.begin(), ids_b.end());
  vector<long> am(ttis.size(), 1);
  torch::Tensor token_ids =
      torch::from_blob(ids_a.data(), {(long)max_length}, opts_data);
  torch::Tensor attention_mask =
      torch::from_blob(am.data(), {(long)max_length}, opts_data);
  torch::Tensor token_type_ids =
      torch::from_blob(ttis.data(), {(long)max_length}, opts_data);
  torch::Tensor position_ids = torch::arange(0, (long)max_length, opts_data);
  torch::Tensor dummy_label = torch::zeros(1, opts_data);
  TransformerFeatures<> features = {token_ids, attention_mask, token_type_ids,
                                    position_ids, dummy_label};
  return features;
}

template <typename T, typename U>
vector<long>
TokenizerBase::create_token_type_ids_from_sequences(vector<T> &tokens_vec_0,
                                                    vector<U> &tokens_vec_1) {
  vector<long> ttis_a(tokens_vec_0.size(), 0);
  vector<long> ttis_b(tokens_vec_1.size(), 1);
  ttis_a.insert(ttis_a.end(), ttis_b.begin(), ttis_b.end());
  return ttis_a;
}

vector<string> TokenizerBase::convert_ids_to_tokens(vector<long> &ids) {
  vector<string> tokens;
  return tokens;
}

string TokenizerBase::convert_tokens_to_string(vector<string> &tokens) {
  string text;
  return text;
}

string TokenizerBase::decode(vector<long> &token_ids) {
  string text;
  return text;
}

void TokenizerBase::truncate_sequences(
    vector<int> &ids_int_a, vector<int> &ids_int_b, size_t num_tokens_to_remove,
    const char *truncation_strategy = "longest_first", size_t stride = 0) {
  if (strcmp(truncation_strategy, "longest_first") == 0) {
    while (num_tokens_to_remove > 0) {
      if (ids_int_b.empty() || ids_int_a.size() > ids_int_b.size()) {
        ids_int_a.pop_back();
        --num_tokens_to_remove;
      } else {
        ids_int_b.pop_back();
      }
    }
  } else if (strcmp(truncation_strategy, "only_first") == 0) {
    while (num_tokens_to_remove > 0 && !ids_int_a.empty()) {
      ids_int_a.pop_back();
      --num_tokens_to_remove;
    }
  } else if (strcmp(truncation_strategy, "only_second") == 0) {
    while (num_tokens_to_remove > 0 && !ids_int_b.empty()) {
      ids_int_b.pop_back();
      --num_tokens_to_remove;
    }
  } else if (strcmp(truncation_strategy, "only_second") == 0) {
    assert((num_tokens_to_remove == 0));
  } else {
    cerr << "invalid truncation strategy.  skipping trancation" << endl;
  }
}

template vector<int> TokenizerBase::tokenize<int>(string &);
template vector<string> TokenizerBase::tokenize<string>(string &);
template vector<long>
TokenizerBase::create_token_type_ids_from_sequences(vector<int> &,
                                                    vector<long> &);
