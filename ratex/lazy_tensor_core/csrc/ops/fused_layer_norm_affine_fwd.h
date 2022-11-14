/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class FusedLayerNormAffineFwd : public Node {
 public:
  FusedLayerNormAffineFwd(const Value& input, const Value& wgt, const Value& bias, 
    const std::vector<int64_t>& normalized_shape, double eps);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

 private:
  double eps_;
  std::vector<int64_t> normalized_shape_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
