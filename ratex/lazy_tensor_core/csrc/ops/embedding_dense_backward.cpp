/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/embedding_dense_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

EmbeddingDenseBackward::EmbeddingDenseBackward(
    const Value& grad_output, const Value& indices, const int64_t num_weights)
    : Node(ir::OpKind(at::aten::embedding_dense_backward), {grad_output, indices}),
      num_weights_(num_weights) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr EmbeddingDenseBackward::Clone(OpList operands) const {
  return MakeNode<EmbeddingDenseBackward>(operands.at(0), operands.at(1), this->num_weights_);
}

std::string EmbeddingDenseBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString();
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors