#pragma once

#include <torch/data/transforms.h>
#include <vector>

#include "transformer_example.h"

template <>
struct torch::data::transforms::Stack<TransformerFeatures<>>
    : public torch::data::transforms::Collation<TransformerFeatures<>> {
  TransformerFeatures<>
  apply_batch(std::vector<TransformerFeatures<>> examples) override {
    std::vector<torch::Tensor> ids, ams, ttis, ps, lbls;
    ids.reserve(examples.size());
    ams.reserve(examples.size());
    ttis.reserve(examples.size());
    ps.reserve(examples.size());
    lbls.reserve(examples.size());
    for (auto &example : examples) {
      ids.push_back(std::move(example.input_ids));
      ams.push_back(std::move(example.attention_mask));
      ttis.push_back(std::move(example.token_type_ids));
      ps.push_back(std::move(example.position_ids));
      lbls.push_back(std::move(example.label));
    }
    return {torch::stack(ids), torch::stack(ams), torch::stack(ttis),
            torch::stack(ps), torch::stack(lbls)};
  }
};
