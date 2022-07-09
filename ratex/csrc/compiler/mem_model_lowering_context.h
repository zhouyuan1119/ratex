/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_set>

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensor_core/csrc/tensor.h"

namespace torch_lazy_tensors {
namespace compiler {
namespace mem_model_lowering_backend {

/*! \brief Dummy computation class to satisfy interface requirements. */
class GenericComputationMemModel : public lazy_tensors::GenericComputation {
 public:
  GenericComputationMemModel(const std::vector<LazyTensor>& tensors, 
                             const std::vector<const ir::Node*> post_order,
                             const std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data)
  : tensors_(tensors), post_order_(post_order), parameters_data_(parameters_data) {}

  /*! \brief Returns the type of the function, i.e. (param_0_ty, param_1_ty, ...) -> ret_ty */
  lazy_tensors::StatusOr<lazy_tensors::ProgramShape> GetProgramShape() const override;

  // Interface functions to read private members
  std::vector<LazyTensor> GetTensors() {
    return tensors_;
  }

  std::vector<const ir::Node*> GetPostOrderNodes() {
    return post_order_;
  }

  std::vector<lazy_tensors::ComputationClient::DataPtr> GetParamsData() {
    return parameters_data_;
  }

 private:
  /*! \brief The list of tensors in this computation */
  std::vector<LazyTensor> tensors_;
  /*! \brief Operators (IR nodes) in topological order */
  std::vector<const ir::Node*> post_order_;
  /*! \brief Parameter information */
  std::vector<lazy_tensors::ComputationClient::DataPtr> parameters_data_;

};

// class MemModelLoweringContext : public ir::LoweringContext {
//  public:
//   MemModelLoweringContext(const std::string& name, Device device) : ir::LoweringContext(name, device) {
//   }
// 
//   MemModelLoweringContext(const std::string& name, Device device,
//                      absl::Span<const ir::Node* const> post_order,
//                      ir::Util::EmissionMap emit_status)
//       : ir::LoweringContext(name, device, post_order, emit_status) {
//     auto lowering = NodeLowering::Create(this);
//     for (auto node : post_order) {
//       bool ok = lowering->Lower(node);
//       LTC_CHECK(ok) << "Failed to lower: " << *node;
//     }
//   }
// 
//   lazy_tensors::Shape GetResultShape(size_t index) const override;
// 
//   size_t AddResult(const ir::Output& output) override;
// 
//   lazy_tensors::StatusOr<std::shared_ptr<lazy_tensors::GenericComputation>> Build() override;
// 
//   void LowerNodeToResult(const ir::Node* node) override;
// 
//   // void AddParameter(const ir::Output& output, size_t index,
//   //                   const lazy_tensors::Shape& shape,
//   //                   const std::string& name) override;
// 
//   void SetUpAlias(const lazy_tensors::ShapeIndex& output_index, int64_t param_number,
//                   const lazy_tensors::ShapeIndex& param_index) override;
// 
// };

/*! \brief A completely dummy function to satisfy interface requirements. */
// ir::Node* LowerNodeForMemModel(const ir::Node* node, MemModelLoweringContext* loctx);

}  // namespace raf_backend
}  // namespace compiler
}  // namespace torch_lazy_tensors
