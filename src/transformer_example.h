#pragma once

#include <torch/types.h>
#include <string>

using namespace std;

struct InputExample {
  string guid;
  string text_a;
  string text_b;
  string label;
};

struct TransformersExample {
  torch::Tensor token_ids;
  torch::Tensor attention_mask;
  torch::Tensor token_type_ids;
  torch::Tensor position_ids;
};
