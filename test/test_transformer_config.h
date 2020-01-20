#include "src/config_utils.h"

TEST(transformerconfigTest, read_pretrained) {
  const char *pretrained_dir =
      "/home/david/Programming/experiments/misc/transformers/bert";
  auto configs = read_transformers_pretrained(pretrained_dir);
  EXPECT_EQ(configs.tokenizer_config.do_lower_case, true);
  EXPECT_EQ(configs.tokenizer_config.init_inputs, vector<string>({}));
  EXPECT_EQ(configs.tokenizer_config.max_len, 512);
  EXPECT_EQ(configs.special_tokens_map.cls_token, "[CLS]");
  EXPECT_EQ(configs.special_tokens_map.mask_token, "[MASK]");
  EXPECT_EQ(configs.special_tokens_map.pad_token, "[PAD]");
  EXPECT_EQ(configs.special_tokens_map.sep_token, "[SEP]");
  EXPECT_EQ(configs.special_tokens_map.unk_token, "[UNK]");
  EXPECT_EQ(configs.added_tokens.added_tokens.size(), 0);
}
