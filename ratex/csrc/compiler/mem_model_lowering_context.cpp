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

/* Member functions of GenericComputationMemModel */

lazy_tensors::StatusOr<lazy_tensors::ProgramShape> GenericComputationMemModel::GetProgramShape() const {
  std::vector<lazy_tensors::Shape> parameter_shapes;
  lazy_tensors::Shape result_shape;
  std::vector<std::string> parameter_names;

  int64_t param_idx = 0;
  for (auto param : parameters_) {
    parameter_shapes.push_back(param->shape());
    parameter_names.push_back(absl::StrCat("p", param_idx));
    param_idx ++;
  }

  if (outputs_.size() > 1) {
    std::vector<lazy_tensors::Shape> fields;
    for (auto output : outputs_) {
      fields.push_back(output->shape());
    }
    result_shape = lazy_tensors::Shape(fields);
  } else {
    result_shape = outputs_[0]->shape();
  }
  
  return lazy_tensors::ProgramShape(parameter_shapes, parameter_names, result_shape);
}

/* Member functions of MemModelLoweringContext */

size_t MemModelLoweringContext::AddResult(const ir::Output& output) {
  outputs_.push_back(output.node);
  return outputs_.size();
}

lazy_tensors::Shape MemModelLoweringContext::GetResultShape(size_t index) const {
  LTC_CHECK(outputs_.size() > index) << "Requesting output shape of tensor " << index
    << ", while we currently only have " << outputs_.size() << " outputs!";
  return outputs_.at(index)->shape();
}

void MemModelLoweringContext::LowerNodeToResult(const ir::Node* node) {
  LTC_LOG(FATAL) << "Not implemented since this seems to be only used in op-by-op mode!";
}

lazy_tensors::StatusOr<std::shared_ptr<lazy_tensors::GenericComputation>> 
MemModelLoweringContext::Build() {
  std::shared_ptr<lazy_tensors::GenericComputation> computation(
    std::make_shared<GenericComputationMemModel>(nodes_, parameters_nodes_, outputs_, alias_, param_alias_));
  return computation;
}

void MemModelLoweringContext::SetUpAlias(
  const lazy_tensors::ShapeIndex& output_index, int64_t param_number,
  const lazy_tensors::ShapeIndex& param_index) {
  LTC_CHECK_EQ(output_index.size(), 1U);
  LTC_CHECK_EQ(param_index.size(), 0U);
  // Aliasing multiple outputs to one parameter is unlikely
  LTC_CHECK(alias_.find(output_index[0]) == alias_.end());
  alias_[output_index[0]] = param_number;
}


}
}

namespace ir {

std::unique_ptr<LoweringContext> LoweringContext::Create(
    const std::string& name, Device device, lazy_tensors::Span<const Node* const> post_order,
    Util::EmissionMap emit_status) {
  return std::make_unique<compiler::mem_model_lowering_backend::MemModelLoweringContext>(
    name, device, post_order, emit_status);
}

std::unique_ptr<LoweringContext> LoweringContext::Create(const std::string& name, Device device) {
  return std::make_unique<compiler::mem_model_lowering_backend::MemModelLoweringContext>(name, device);
}

}  // namespace ir

}

