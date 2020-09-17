#include "tokenizer_albert.h"

using namespace std;

namespace hflt {

shared_ptr<sentencepiece::SentencePieceProcessor>
load_spmodel(const char *sppath) {
  shared_ptr<sentencepiece::SentencePieceProcessor> sp(
      new sentencepiece::SentencePieceProcessor());
  sp->LoadOrDie(sppath);
  return sp;
}

string get_spmodel_path(const char *pretrained_dir) {
  string s(pretrained_dir);
  s += "/spiece.model";
  return s;
}

TokenizerAlbert::TokenizerAlbert(const char *pretrained_dir)
    : TokenizerAlbert(read_transformers_pretrained(pretrained_dir),
                      get_spmodel_path(pretrained_dir).c_str()){};

TokenizerAlbert::TokenizerAlbert(TransformersTokenizerConfigs configs,
                                 const char *spmodel_path)
    : TokenizerBase(configs.special_tokens_map.bos_token,
                    configs.special_tokens_map.eos_token,
                    configs.special_tokens_map.unk_token,
                    configs.special_tokens_map.sep_token,
                    configs.special_tokens_map.pad_token,
                    configs.special_tokens_map.cls_token,
                    configs.special_tokens_map.mask_token,
                    configs.added_tokens.added_tokens, 0),
      processor_(load_spmodel(spmodel_path)){};

TransformerFeatures<> TokenizerAlbert::encode(string &text_a, string &text_b,
                                              bool add_special_tokens,
                                              size_t max_length, size_t stride,
                                              const char *truncation_strategy,
                                              bool pad_to_max_length) {
  // create a type for torch tensors
  auto opts_data = torch::TensorOptions().dtype(torch::kLong);
  // encode texts to ids directly
  vector<int> tis_int_a;
  vector<int> tis_int_b;
  processor_->Encode(text_a, &tis_int_a);
  processor_->Encode(text_b, &tis_int_b);
  // truncate based on length and strategy
  size_t num_tokens = tis_int_a.size() + 2;
  if (!tis_int_b.empty()) {
    num_tokens += tis_int_b.size() + 1;
  }
  truncate_sequences(tis_int_a, tis_int_b,
                     min((size_t)0, num_tokens - max_length),
                     truncation_strategy, stride);
  // add in special tokens
  tis_int_a.insert(tis_int_a.begin(), processor_->PieceToId(cls_token));
  tis_int_a.emplace_back(processor_->PieceToId(sep_token));
  if (!tis_int_b.empty()) {
    tis_int_b.emplace_back(processor_->PieceToId(sep_token));
  }
  // convert from int (sentencepiece type) to long (torch type)
  vector<long> tis;
  tis.reserve(tis_int_a.size() + tis_int_b.size());
  tis.insert(tis.end(), tis_int_a.begin(), tis_int_a.end());
  tis.insert(tis.end(), tis_int_b.begin(), tis_int_b.end());
  tis.resize(max_length, processor_->PieceToId(pad_token));
  // create attention_mask, token_type_ids, and position_ids
  vector<long> am(tis_int_a.size() + tis_int_b.size(), 1);
  am.resize(max_length, 0);
  vector<long> ttis(tis_int_a.size(), 0);
  ttis.resize(max_length, 1);
  // convert from standard data types to torch tensors
  torch::Tensor token_ids =
      torch::from_blob(tis.data(), {(long)max_length}, opts_data).clone();
  torch::Tensor attention_mask =
      torch::from_blob(am.data(), {(long)max_length}, opts_data).clone();
  torch::Tensor token_type_ids =
      torch::from_blob(ttis.data(), {(long)max_length}, opts_data).clone();
  torch::Tensor position_ids = torch::arange(0, (long)max_length, opts_data);
  // add a dummy label to the features, replaced later
  torch::Tensor dummy_label = torch::zeros({1}, opts_data);
  // create struct of features which is returned by the dataset
  TransformerFeatures<> features = {token_ids, attention_mask, token_type_ids,
                                    position_ids, dummy_label};
  return features;
}

template <typename T> vector<T> TokenizerAlbert::tokenize(string &text) {
  vector<T> tokens;
  processor_->Encode(text, &tokens);
  return tokens;
}

string TokenizerAlbert::decode(vector<long> &token_ids) {
  vector<int> tokens_int(token_ids.begin(), token_ids.end());
  string text;
  processor_->Decode(tokens_int, &text);
  return text;
}

int TokenizerAlbert::cls_token_id() { return processor_->PieceToId(cls_token); }

int TokenizerAlbert::sep_token_id() { return processor_->PieceToId(sep_token); }

int TokenizerAlbert::pad_token_id() { return processor_->PieceToId(pad_token); }

template vector<int> TokenizerAlbert::tokenize(string &);
template vector<string> TokenizerAlbert::tokenize(string &);

}; // namespace hflt
