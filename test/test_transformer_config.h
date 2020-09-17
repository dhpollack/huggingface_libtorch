#pragma once

#include "catch2/catch.hpp"
#include "src/config_utils.h"

using namespace hflt;

TEST_CASE("Test Transformer Read Config", "[config]") {
  const char *pretrained_dir = "models/sst2_trained";
  auto configs = read_transformers_pretrained(pretrained_dir);
  REQUIRE(configs.tokenizer_config.do_lower_case == false);
  REQUIRE(configs.tokenizer_config.init_inputs == std::vector<std::string>({}));
  REQUIRE(configs.tokenizer_config.max_len == 512);
  REQUIRE(configs.special_tokens_map.cls_token == "[CLS]");
  REQUIRE(configs.special_tokens_map.mask_token == "[MASK]");
  REQUIRE(configs.special_tokens_map.pad_token == "<pad>");
  REQUIRE(configs.special_tokens_map.sep_token == "[SEP]");
  REQUIRE(configs.special_tokens_map.unk_token == "<unk>");
  REQUIRE(configs.added_tokens.added_tokens.size() == 0);
}
