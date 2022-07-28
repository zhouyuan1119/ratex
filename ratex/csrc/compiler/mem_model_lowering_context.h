/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_set>

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensor_core/csrc/tensor.h"
#include "client/base_computation_client.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"

namespace torch_lazy_tensors {
namespace compiler {
namespace mem_model_lowering_backend {

/*! \brief Dummy computation class to satisfy interface requirements. */
class GenericComputationMemModel : public lazy_tensors::GenericComputation {
 public:
  GenericComputationMemModel(
    const std::vector<const ir::Node*>& nodes,
    const std::vector<const ir::Node*>& parameters,
    const std::vector<const ir::Node*>& outputs,
    const std::unordered_map<int64_t, int64_t>& alias,
    const std::unordered_map<const ir::Node*, int64_t>& param_alias)
  : nodes_(nodes), parameters_(parameters), outputs_(outputs), alias_(alias), param_alias_(param_alias) {}

  /*! \brief Returns the type of the function, i.e. (param_0_ty, param_1_ty, ...) -> ret_ty */
  lazy_tensors::StatusOr<lazy_tensors::ProgramShape> GetProgramShape() const override;

  /*! \brief Some interface functions to retrieve the private members */
  const std::vector<const ir::Node*> GetPostOrderNodes() { return nodes_; }
  const std::unordered_map<int64_t, int64_t> GetAlias() { return alias_; }
  const std::vector<const ir::Node*> GetOutputs() { return outputs_; }
  const std::vector<const ir::Node*> GetParameters() { return parameters_; };
  const std::unordered_map<const ir::Node*, int64_t> GetParamAlias() { return param_alias_; }

 private:
  /*! \brief A list of nodes, sorted in topological order. */
  std::vector<const ir::Node*> nodes_;
  /*! \brief Collection of parameter nodes. Maps from handles to ir nodes. */
  std::vector<const ir::Node*> parameters_;
  /*! \brief A vector of outputs. We treat all live tensors as outputs. */
  std::vector<const ir::Node*> outputs_;
  /*! \brief Maps output to input if they are aliased */
  std::unordered_map<int64_t, int64_t> alias_;
  /*! \brief Maps a parameter node to another parameter if they are aliased */
  std::unordered_map<const ir::Node*, int64_t> param_alias_;
};

class MemModelLoweringContext : public ir::LoweringContext {
 public:
  MemModelLoweringContext(const std::string& name, Device device) : ir::LoweringContext(name, device) {
  }

  MemModelLoweringContext(const std::string& name, Device device,
                     absl::Span<const ir::Node* const> post_order,
                     ir::Util::EmissionMap emit_status)
      : ir::LoweringContext(name, device, post_order, emit_status) {
    // Sort the post order nodes in-place here
    // It is not used in the constructor of LoweringContext
    nodes_.insert(nodes_.end(), post_order.begin(), post_order.end());
    std::sort(nodes_.begin(), nodes_.end(), CompareNodes);
    std::unordered_map<lazy_tensors::client::Data::OpaqueHandle, size_t> data_handles;
    for (int64_t i = 0; i < nodes_.size(); i ++) {
      auto node = nodes_[i];
      // nodes_.push_back(node);
      // Collect all parameter nodes
      if (node->op() == *torch_lazy_tensors::ir::ops::ltc_device_data) {
        auto device_data_node = ir::NodeCast<ir::ops::DeviceData>(node, *ir::ops::ltc_device_data);
        lazy_tensors::client::Data::OpaqueHandle handle = device_data_node->data()->GetOpaqueHandle();
        auto it = data_handles.find(handle);
        if (it == data_handles.end()) {
          // Add the node into parameters
          parameters_.push_back(device_data_node->data());
          data_handles[handle] = parameters_nodes_.size();
          parameters_nodes_.push_back(node);
        } else {
          auto aliased_param_id = data_handles.at(handle);
          param_alias_.insert(std::make_pair(node, aliased_param_id));
        }
      }
    }
  }

  /*! 
   * \brief Get the output shape of the tensor at the provided index. 
   * e.g., GetResultShape(3) would return the shape of the 3rd output tensor. 
   * The order is the same with the order these tensors are added via AddResult(). 
   */
  lazy_tensors::Shape GetResultShape(size_t index) const override;

  /*! \brief Add one output to the lowering context. We maintain all outputs in a vector. */
  size_t AddResult(const ir::Output& output) override;

  /*! \brief Generate a GenericComputationMemModel from this lowering context. */
  lazy_tensors::StatusOr<std::shared_ptr<lazy_tensors::GenericComputation>> Build() override;

  // This function is not implemented, do not use
  void LowerNodeToResult(const ir::Node* node) override;

  /* We don't implement the GetOutputOp(), AssignOutputOp(), and GetParameter() methods as in 
   * RAFLoweringContext because we don't actually do any lowering. */

  void SetUpAlias(const lazy_tensors::ShapeIndex& output_index, int64_t param_number,
                  const lazy_tensors::ShapeIndex& param_index) override;

 private:
  /*! \brief A utility for properly sorting the IR nodes. */
  static bool CompareNodes(const ir::Node* const node0, const ir::Node* const node1) {
    return node0->id() < node1->id();
  }

  /*! \brief A list of nodes, sorted in topological order. */
  std::vector<const ir::Node*> nodes_;
  /*! \brief A list of parameter nodes. */
  std::vector<const ir::Node*> parameters_nodes_;
  /*! \brief Collection of model states (weights). */
  std::unordered_set<const ir::Node*> model_states_;
  /*! \brief A vector of outputs. We treat all live tensors as outputs. */
  std::vector<const ir::Node*> outputs_;
  /*! \brief maps output to input if they are aliased */
  std::unordered_map<int64_t, int64_t> alias_;
  /*! \brief map to record parameter aliasing */
  std::unordered_map<const ir::Node*, int64_t> param_alias_;
};

}  // namespace raf_backend
}  // namespace compiler
}  // namespace torch_lazy_tensors
