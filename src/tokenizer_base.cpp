#include "tokenizer_base.h"

TokenizerBase::TokenizerBase(string bos_token, string eos_token,
			     string unk_token, string sep_token,
			     string pad_token, string cls_token,
			     string mask_token,
			     vector<string> additional_special_tokens,
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

vector<string> TokenizerBase::tokenize(string &text) {
    vector<string> tokens;
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

tpenc TokenizerBase::encode(string &text_a, string &text_b,
			    bool add_special_tokens = true,
			    size_t max_length = 512, size_t stride = 0,
			    const char *truncation_strategy = "longest_first",
			    bool pad_to_max_length = false) {
    vector<string> tokens_a = tokenize(text_a);
    // TODO add special tokens here
    vector<long> ids_a = convert_tokens_to_ids(tokens_a);
    vector<string> tokens_b = tokenize(text_b);
    // TODO add special tokens here
    vector<long> ids_b = convert_tokens_to_ids(tokens_b);
    // TODO truncate ids here
    vector<long> ttis = create_token_type_ids_from_sequences(ids_a, ids_b);
    ids_a.insert(ids_a.end(), ids_b.begin(), ids_b.end());
    vector<long> am(ttis.size(), 1);
    // TODO pad here
    tpenc tup_itm = make_tuple(ids_a, ttis, am);
    return tup_itm;
}

vector<long>
TokenizerBase::create_token_type_ids_from_sequences(vector<long> &token_ids_0,
						    vector<long> &token_ids_1) {
    vector<long> ttis_a(token_ids_0.size(), 0);
    vector<long> ttis_b(token_ids_1.size(), 1);
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

