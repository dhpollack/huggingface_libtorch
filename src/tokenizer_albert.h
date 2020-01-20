#pragma once

#include "sentencepiece_processor.h"
#include "tokenizer_base.h"
#include "config_utils.h"

class TokenizerAlbert : TokenizerBase {
public:
  TokenizerAlbert(const char *pretrained_dir);
  TokenizerAlbert(TransformersTokenizerConfigs configs, const char *spmodel_path);
private:
  std::shared_ptr<sentencepiece::SentencePieceProcessor> processor_;
};
