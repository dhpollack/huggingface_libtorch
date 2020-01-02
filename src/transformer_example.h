#pragma once

#include <torch/types.h>

namespace torch {
namespace data {

/// A `TransformerExample` from a dataset.
///
/// A dataset consists of data and an associated target (label).
template <typename InputID = Tensor, 
	  typename AttentionMask = Tensor, 
	  typename TokenTypeID = Tensor,
	  typename PositionID = Tensor,
	  typename Target = Tensor>
struct TransformerExample {
  using InputIDType = InputID;
  using AttentionMaskType = AttentionMask;
  using TokenTypeIDType = TokenTypeID;
  using PositionIDType = PositionID;
  using TargetType = Target;

  TransformerExample() = default;
  TransformerExample(InputID input_ids, 
                     AttentionMask attention_mask, 
	             TokenTypeID token_type_ids, 
	             PositionID position_ids,
	             Target target)
      : input_ids(std::move(input_ids)),
        attention_mask(std::move(attention_mask)),
        token_type_ids(std::move(token_type_ids)),
        position_ids(std::move(position_ids)),	
	target(std::move(target)) {}

  InputID input_ids;
  AttentionMask attention_mask;
  TokenTypeID token_type_ids;
  PositionID position_ids;
  Target target;
};

}
}
