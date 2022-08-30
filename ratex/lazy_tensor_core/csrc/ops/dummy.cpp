/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/dummy.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Dummy::Dummy(const Value& input)
    : Node(ltc_dummy, {input}) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Dummy::Clone(OpList operands) const {
  return MakeNode<Dummy>(operands.at(0));
}

std::string Dummy::ToString() const {
  std::stringstream ss;
  ss << Node::ToString();
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors