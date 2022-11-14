/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/fused_layer_norm_affine_bwd.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

FusedLayerNormAffineBwd::FusedLayerNormAffineBwd(const Value& grad_output, 
    const Value& mean, const Value& invvar, const Value& input, const Value& wgt, 
    const Value& bias, const std::vector<int64_t>& normalized_shape, double eps)
    : Node(ltc_fused_layer_norm_affine_bwd, {grad_output, mean, invvar, input, wgt, bias}, 3), normalized_shape_(normalized_shape), eps_(eps) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr FusedLayerNormAffineBwd::Clone(OpList operands) const {
  return MakeNode<FusedLayerNormAffineBwd>(
      operands.at(0), operands.at(1), operands.at(2), operands.at(3), operands.at(4), operands.at(5), 
      normalized_shape_, eps_);
}

std::string FusedLayerNormAffineBwd::ToString() const {
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