#pragma once

#include <string>
#include <torch/types.h>

namespace hflt {

struct TransformerExample {
  std::string guid;
  std::string text_a;
  std::string text_b;
  std::string label;

  TransformerExample() = default;
  TransformerExample(std::string guid, std::string text_a, std::string text_b,
                     std::string label)
      : guid(guid), text_a(text_a), text_b(text_b), label(label){};
};

/// A `TransformerFeatures` from a dataset.
///
/// A dataset consists of data and an associated target (label).
template <typename InputID = torch::Tensor,
          typename AttentionMask = torch::Tensor,
          typename TokenTypeID = torch::Tensor,
          typename PositionID = torch::Tensor, typename Label = torch::Tensor>
struct TransformerFeatures {
  using InputIDType = InputID;
  using AttentionMaskType = AttentionMask;
  using TokenTypeIDType = TokenTypeID;
  using PositionIDType = PositionID;
  using LabelType = Label;

  TransformerFeatures() = default;
  TransformerFeatures(InputID input_ids, AttentionMask attention_mask,
                      TokenTypeID token_type_ids, PositionID position_ids,
                      Label label)
      : input_ids(std::move(input_ids)),
        attention_mask(std::move(attention_mask)),
        token_type_ids(std::move(token_type_ids)),
        position_ids(std::move(position_ids)), label(std::move(label)) {}

  InputIDType input_ids;
  AttentionMaskType attention_mask;
  TokenTypeIDType token_type_ids;
  PositionIDType position_ids;
  LabelType label;
};

}; // namespace hflt
