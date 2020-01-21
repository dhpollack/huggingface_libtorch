#include "tokenizer_albert.h"

using namespace std;

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
      processor_(move(load_spmodel(spmodel_path))){};

tpenc TokenizerAlbert::encode(string &text_a, string &text_b,
			      bool add_special_tokens, size_t max_length,
			      size_t stride, const char *truncation_strategy,
			      bool pad_to_max_length) {
  vector<int> token_ids_int_a;
  vector<int> token_ids_int_b;
  processor_->Encode(text_a, &token_ids_int_a);
  processor_->Encode(text_b, &token_ids_int_b);
  // TODO the sp model does not encode the CLS and SEP tokens
  token_ids_int_a.insert(token_ids_int_a.begin(), processor_->PieceToId(cls_token));
  token_ids_int_a.push_back(processor_->PieceToId(sep_token));
  if (token_ids_int_b.size() > 0)
    token_ids_int_b.push_back(processor_->PieceToId(sep_token));
  vector<long> token_ids;
  //token_ids.reserve(token_ids_int_a.size() + token_ids_int_b.size());
  token_ids.insert(token_ids.end(), token_ids_int_a.begin(),
  		   token_ids_int_a.end());
  token_ids.insert(token_ids.end(), token_ids_int_b.begin(),
		   token_ids_int_b.end());
  token_ids.resize(max_length, 0);
  vector<long> attention_mask(token_ids_int_a.size() + token_ids_int_b.size(),
			      1);
  attention_mask.resize(max_length, 0);
  vector<long> token_type_ids(token_ids_int_a.size(), 0);
  //token_type_ids.resize(token_ids_int_a.size() + token_ids_int_b.size(), 1);
  token_type_ids.resize(max_length, 1);
  return make_tuple(token_ids, attention_mask, token_type_ids);
}
