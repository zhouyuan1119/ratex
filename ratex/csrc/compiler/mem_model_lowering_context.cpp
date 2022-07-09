/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/compiler/mem_model_lowering_context.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensors/computation_client/computation_client.h"
#include "client/base_computation_client.h"
#include "lazy_tensors/shape.h"

#include "./utils.h"

namespace torch_lazy_tensors {
namespace compiler {
namespace mem_model_lowering_backend {

lazy_tensors::StatusOr<lazy_tensors::ProgramShape> GenericComputationMemModel::GetProgramShape() const {
  std::vector<lazy_tensors::Shape> parameter_shapes;
  lazy_tensors::Shape result_shape;
  std::vector<std::string> parameter_names;

  for (int64_t idx = 0; idx < parameters_data_.size(); idx ++) {
    parameter_shapes.push_back(lazy_tensors::Shape(parameters_data_[idx]->shape()));
    parameter_names.push_back(absl::StrCat("p", idx));
  }

  // Get result shape from the topologically sorted nodes
  // Assuming the last node is the output, subject to change
  auto output_node = post_order_.back();
  result_shape = output_node->shape();
  
  return lazy_tensors::ProgramShape(parameter_shapes, parameter_names, result_shape);
}

}
}
}

