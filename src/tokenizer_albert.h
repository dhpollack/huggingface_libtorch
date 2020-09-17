#pragma once
#include <sentencepiece_processor.h>

#include "config_utils.h"
#include "tokenizer_base.h"
#include "transformer_example.h"

namespace hflt {

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

}; // namespace hflt
