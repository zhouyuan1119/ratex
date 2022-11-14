/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/fused_layer_norm_affine_fwd.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

FusedLayerNormAffineFwd::FusedLayerNormAffineFwd(const Value& input, const Value& wgt, 
  const Value& bias, const std::vector<int64_t>& normalized_shape, double eps)
    : Node(ltc_fused_layer_norm_affine_fwd, {input, wgt, bias}, 3), normalized_shape_(normalized_shape), eps_(eps) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr FusedLayerNormAffineFwd::Clone(OpList operands) const {
  return MakeNode<FusedLayerNormAffineFwd>(operands.at(0), operands.at(1), operands.at(2), normalized_shape_, eps_);
}

std::string FusedLayerNormAffineFwd::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", normalized_shape = [ ";
  for (auto elm : normalized_shape_) {
      ss << elm << " ";
  }
  ss << "], eps = " << eps_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors