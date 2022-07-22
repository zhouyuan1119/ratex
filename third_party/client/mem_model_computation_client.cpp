/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "client/mem_model_computation_client.h"

#include <fstream>
#include <iostream>

#include "ratex/csrc/compiler/utils.h"
#include "ratex/csrc/compiler/mem_model_lowering_context.h"
#include "ratex/csrc/value_ext/value.h"
#include "ratex/csrc/pass_ext/pass.h"
#include "ratex/csrc/utils/file.h"
#include "env_vars.h"

#include "lazy_tensors/computation_client/nnc_computation_client.h"
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"

namespace ratex {

using namespace torch_lazy_tensors::compiler;
using namespace torch_lazy_tensors::compiler::mem_model_lowering_backend;

std::unique_ptr<ComputationClient> MemModelComputationClient::Create() {
  Options options;
  PopulateLocalDevices(&options);
  return std::make_unique<MemModelComputationClient>(options);
}

ComputationClient::DataPtr MemModelComputationClient::CreateDataPlaceholder(std::string device, Shape shape) {
  return DataPtr(std::make_shared<MemModelData>(std::move(device), shape));   
}

std::vector<ComputationClient::DataPtr> MemModelComputationClient::TransferToServer(
    lazy_tensors::Span<const TensorSource> tensors) {
  std::vector<ComputationClient::DataPtr> result;
  for (const auto& ts : tensors) {
    result.push_back(DataPtr(std::make_shared<MemModelData>(ts.device, Shape(ts.shape))));
  }
  return result;
}

std::vector<Literal> MemModelComputationClient::TransferFromServer(
    lazy_tensors::Span<const DataPtr> handles) {
  std::vector<Literal> results;
  for (const auto& handle : handles) {
    auto* ptr = static_cast<BaseData*>(handle.get());
    LTC_LOG(INFO) << "TransferFromServer: shape = " << Shape(ptr->shape()).ToString();
    Literal res(ptr->shape());
    LTC_CHECK(ptr->shape().element_shapes().size() == 0) << "Tuple is not supported!";
    auto dtype = ptr->shape().element_type();
    // If this tensor is a parameter, then we don't allocate memory for it because it should
    // already bound to some memory. We are using new here and may have some memory leaks. To 
    // be fixed later. 
    if (!ptr->is_param) {
      int64_t n_elements = res.value().numel();
      switch(dtype) {
        case PrimitiveType::S8: {
          const int8_t* buf = new int8_t[n_elements]();
          res.PopulateR1<int8_t>(Span<const int8_t>(buf, n_elements));
          break;
        }
        case PrimitiveType::U8: {
          const uint8_t* buf = new uint8_t[n_elements]();
          res.PopulateR1<uint8_t>(Span<const uint8_t>(buf, n_elements));
          break;
        }
        case PrimitiveType::PRED: {
          const bool* buf = new bool[n_elements]();
          res.PopulateR1<bool>(Span<const bool>(buf, n_elements));
          break;
        }
        case PrimitiveType::S32: {
          const int32_t* buf = new int32_t[n_elements]();
          res.PopulateR1<int32_t>(Span<const int32_t>(buf, n_elements));
          break;
        }
        case PrimitiveType::U32: {
          const uint32_t* buf = new uint32_t[n_elements]();
          res.PopulateR1<uint32_t>(Span<const uint32_t>(buf, n_elements));
          break;
        }
        case PrimitiveType::F32: {
          const float* buf = new float[n_elements]();
          res.PopulateR1<float>(Span<const float>(buf, n_elements));
          break;
        }
        case PrimitiveType::S64: {
          const int64_t* buf = new int64_t[n_elements]();
          res.PopulateR1<int64_t>(Span<const int64_t>(buf, n_elements));
          break;
        }
        case PrimitiveType::U64: {
          const uint64_t* buf = new uint64_t[n_elements]();
          res.PopulateR1<uint64_t>(Span<const uint64_t>(buf, n_elements));
          break;
        }
        case PrimitiveType::F64: {
          const double* buf = new double[n_elements]();
          res.PopulateR1<double>(Span<const double>(buf, n_elements));
          break;
        }
        default:
          LTC_LOG(FATAL) << "NotImplementedError: " << dtype;
      }
    }
    results.push_back(res);
  }
  return results;
}

ComputationClient::ComputationPtr MemModelComputationClient::Compile(
    ComputationClient::CompileInstance instance) {
  LTC_LOG(INFO) << "In MemModelComputationClient::Compile";
  auto* computation = static_cast<GenericComputationMemModel*>(instance.computation.get());
  LTC_LOG(INFO) << "Got computation!";

  auto post_order_nodes = computation->GetPostOrderNodes();
  auto alias = computation->GetAlias();
  auto outputs = computation->GetOutputs();
  std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t> outputs_map;
  for (int64_t i = 0; i < outputs.size(); i ++) {
    outputs_map.insert(std::make_pair(outputs[i], i));
  }
  auto params = computation->GetParameters();
  LTC_LOG(INFO) << "Got info!";

  // Walk the graph and get the use count of each node. 
  // We cannot leverage the use count in lazy tensor IR because over there the
  // uses are maintained in a set, which will cause issues for our analysis. 
  auto use_cnts = AnalyzeUseCount(post_order_nodes);
  LTC_LOG(INFO) << "Got use counts!";

  // Collect information for correctly calculating memory with in-place updates
  // auto param_tensor_ids = GetParameterTensorIds(params);
  // LTC_LOG(INFO) << "Got parameter ids!";
  // auto node_tensor_map = GetNodeTensorIdMap(tensors);
  // LTC_LOG(INFO) << "Got node tensor id map!";

  // Analyze the graph and build the mem model. 
  double peak_mem_mbs = CalculatePeakMem(outputs_map, 
                                         post_order_nodes,
                                         params,
                                         alias,
                                         use_cnts);
  peak_memory_ = peak_mem_mbs;

  auto ret = std::make_shared<MemModelComputation>(instance.computation,
                                                   ConsumeValue(instance.computation->GetProgramShape()),
                                                   instance.devices, peak_mem_mbs);
  return ret;
}

std::vector<ComputationClient::DataPtr> MemModelComputationClient::ExecuteComputation(
    const Computation& computation, lazy_tensors::Span<const DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  return {};
}

lazy_tensors::ComputationClient* MemModelGet() {
  using namespace lazy_tensors;
  static auto mem_model_computation_client = MemModelComputationClient::Create();
  return mem_model_computation_client.get();
}

lazy_tensors::ComputationClient* MemModelGetIfInitialized() {
  using namespace lazy_tensors;
  return MemModelGet();
}


std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t> AnalyzeUseCount(
  const std::vector<const torch_lazy_tensors::ir::Node*>& topo_sorted_nodes) {
  std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t> use_cnts;
  for (auto* node : topo_sorted_nodes) {
    use_cnts[node] = 0;
    for (auto pred : node->operands()) {
      const torch_lazy_tensors::ir::Node* pred_node = pred.node;
      LTC_CHECK(use_cnts.count(pred_node)) << "Node " << pred_node->ToString() 
                                           << " does not have use count!";
      use_cnts[pred_node] += 1;
    } 
  }
  return use_cnts;
}

int GetElementSizeInBytes(const PrimitiveType elem_ty) {
  int element_size = 0;
  switch (elem_ty) {
    case PrimitiveType::S8:
    case PrimitiveType::U8: element_size = 1; break;
    case PrimitiveType::S16:
    case PrimitiveType::U16:
    case PrimitiveType::F16:
    case PrimitiveType::BF16: element_size = 2; break;
    case PrimitiveType::S32:
    case PrimitiveType::U32:
    case PrimitiveType::F32: element_size = 4; break;
    case PrimitiveType::S64:
    case PrimitiveType::U64:
    case PrimitiveType::F64:
    case PrimitiveType::C64: element_size = 8; break;
    case PrimitiveType::C128: element_size = 16; break;
    default: LTC_LOG(FATAL) << "Unsupported element type " << elem_ty;
  }
  return element_size;
}

double CalculateMemFromShape(const lazy_tensors::Shape& shape) {
  int64_t size = 0;
  if (shape.tuple_shapes_size() == 0) {
    // Single tensor, non-tuple
    int elem_size = GetElementSizeInBytes(shape.element_type());
    size = elem_size;
    for (int64_t dim : shape.dimensions()) {
      size *= dim;
    }
  } else {
    // Tuple
    for (auto elem_shape : shape.tuple_shapes()) {
      int elem_size = GetElementSizeInBytes(elem_shape.element_type());
      for (int64_t dim : elem_shape.dimensions()) {
        elem_size *= dim;
      }
      size += elem_size;
    }
  }
  
  return size / 1048576.0;
}

bool IsInplaceOp(const c10::Symbol op) {
  // Currently we treat all ops whose names end with an underscore as in-place ops
  std::string op_name = std::string(op.toQualString());
  bool is_inplace = pytorch_inplace_ops.count(op_name) || (op_name.back() == '_');
  if (is_inplace)
    LTC_LOG(INFO) << "Op " << op_name << " is an in-place op!";
  else
    LTC_LOG(INFO) << "Op " << op_name << " is not an in-place op. ";
  return is_inplace;
}

std::unordered_set<int64_t> GetParameterTensorIds(
  const std::vector<lazy_tensors::ComputationClient::DataPtr>& params) {
  std::unordered_set<int64_t> param_tensor_ids;
  for (auto param : params) {
    auto* data_info = dynamic_cast<torch_lazy_tensors::DeviceDataInfo*>(param->info());
    if (data_info != nullptr) {
      param_tensor_ids.insert(data_info->tensor_id);
    }
  }
  return param_tensor_ids;  
}

std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t> GetNodeTensorIdMap(
  const std::vector<torch_lazy_tensors::LazyTensor>& tensors) {
  // SOME TENSORS MAY COME FROM THE PREVIOUS BATCH, LIKE INPUTS, LABELS, ETC. 
  std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t> output_tensor_ids;
  for (auto t : tensors) {
    int64_t tid = t.GetUniqueId();
    auto node = t.GetIrValue().node.get();
    LTC_LOG(INFO) << tid << " " << node;
    LTC_LOG(INFO) << node->ToString();
    if (!output_tensor_ids.count(node)) {
      output_tensor_ids.insert(std::make_pair(node, tid));
    } else {
      LTC_LOG(FATAL) << "Node " << node->ToString() << " has multiple outputs!";
    }
  }
  return output_tensor_ids;
}

bool IsSharingWithParam(const torch_lazy_tensors::ir::Node* node,
                        const std::unordered_set<int64_t>& param_tensor_ids,
                        const std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t>& node_tensor_map) {
  if (node_tensor_map.count(node)) 
    return param_tensor_ids.count(node_tensor_map.at(node));
  return false;
}

double CalculatePeakMem(const std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t>& outputs_map,
                        const std::vector<const torch_lazy_tensors::ir::Node*>& topo_sorted_nodes,
                        const std::vector<const torch_lazy_tensors::ir::Node*>& params,
                        const std::unordered_map<int64_t, int64_t>& alias,
                        const std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t>& use_cnts) {
  struct TensorInfo {
    TensorInfo(double size, int64_t uses, bool param) : size_mbs(size), use_cnt(uses), is_param(param) {}

    // Size of tensor in MBs
    double size_mbs;
    // Use count of this tensor
    int64_t use_cnt;
    // True if the tensor is a parameter or shares storage with a parameter
    bool is_param;
    // True if the tensor has undergone an in-place update and is replaced by a newer version
    bool is_expired = false;
  };

  double curr_mem = 0.0;

  // Maintain the current set of live tensors, not including parameters since they are always live
  std::unordered_map<const torch_lazy_tensors::ir::Node*, TensorInfo> live_tensors;
  // A list of tensors that have reached the end of their lifetime, together with their sizes
  std::vector<std::pair<const torch_lazy_tensors::ir::Node*, double>> to_be_freed;

  // Parameters persist in the memory
  for (auto param : params) {
    // Only include parameters that are used in the graph to avoid counting parameters from the previous batch
    if (use_cnts.count(param) && (use_cnts.at(param) > 0)) {
      double param_mem = CalculateMemFromShape(Shape(param->shape()));
      LTC_LOG(INFO) << "Param: " << param_mem << " MBs";
      curr_mem += param_mem;
      // Insert parameters into the live set, they are never removed
      live_tensors.insert(std::make_pair(param, TensorInfo(param_mem, use_cnts.at(param), true)));
    }
  }

  double peak_mem = curr_mem;

  // Assuming all nodes are sorted in topological order and the ops will be executed exactly in this order
  // NOTICE THAT THE LATTER DOES NOT HOLD RIGHT NOW!!!
  for (auto* node : topo_sorted_nodes) {
    LTC_LOG(INFO) << "Analyzing node " << node->ToString() 
                  << ", uses: " << node->uses().size();
    for (auto use : node->uses()) 
      LTC_LOG(INFO) << use.node->ToString();
    // Step 1: Purge any tensors that can be freed
    for (auto dead_node_with_size : to_be_freed) {
      live_tensors.erase(dead_node_with_size.first);
      curr_mem -= dead_node_with_size.second;
      LTC_LOG(INFO) << "Erase dead node " << dead_node_with_size.first->ToString() << " for " 
                    << dead_node_with_size.second << " MBs memory";
    }
    to_be_freed.clear();

    // Step 2: Add the output of the current op to the live set and update current memory
    double outp_size = CalculateMemFromShape(node->shape());
    LTC_CHECK(use_cnts.count(node)) << "Node " << node->ToString() << " does not have use count!";

    /* 
     * There are several cases here:
     * 1. This node is device_data(), which means the memory is already added when processing parameters. 
     *    In this case we don't do anything. 
     * 2. This node is not device_data(), but the op is an in-place op (e.g., permute). In this case
     *    we don't increment memory, but update the TensorInfo associated with the input to reflect
     *    the use count of the output. Right now we only support in-place ops with one input. 
     * 3. This node is not device_data() nor an in-place op, but the output of the node shares memory 
     *    with a parameter. This is defined by the alias map. In this case we don't increment memory, 
     *    but update the TensorInfo associated with the parameter to reflect the use count of this 
     *    tensor. THIS SHOULD NOT HAVE ISSUES IF THE IR ORDER IS THE SAME WITH PYTORCH EXECUTION ORDER, 
     *    BUT MAY HAVE ISSUES BEFORE WE FIX THAT. 
     * 4. This node is not device_data(), not an in-place op, and the output does not share memory
     *    with a parameter. In this case we add the size of the output to curr_mem, and create a new
     *    entry in live_tensors for this output. 
     */
    bool is_inplace = false;
    if (node->op() != *torch_lazy_tensors::ir::ops::ltc_device_data) {
      // In-place op
      if (IsInplaceOp(node->op().op)) {
        is_inplace = true;
        LTC_CHECK(node->operands().size() == 1) << "In-place ops with more than one inputs are currently not supported!";
        auto pred_node = node->operands()[0].node;
        // Mark the entry of the input as expired
        live_tensors.at(pred_node).is_expired = true;
        LTC_CHECK(outp_size == live_tensors.at(pred_node).size_mbs) << "In-place update but tensor sizes mismatch: "
          << outp_size << " vs. " << live_tensors.at(pred_node).size_mbs;
        // Put a new entry
        live_tensors.insert(
          std::make_pair(node, TensorInfo(outp_size, use_cnts.at(node), live_tensors.at(pred_node).is_param))
        );
      } else {
        // Not in-place op, check for I/O param aliasing
        if (outputs_map.count(node) && alias.count(outputs_map.at(node))) {
          // If there is I/O param aliasing, this is the output and mark the param as expired
          auto param_node = params.at(alias.at(outputs_map.at(node)));
          live_tensors.at(param_node).is_expired = true;
          LTC_CHECK(outp_size == live_tensors.at(param_node).size_mbs) << "I/O aliasing but tensor sizes mismatch: "
            << outp_size << " vs. " << live_tensors.at(param_node).size_mbs;
          // Put a new entry
          live_tensors.insert(
            std::make_pair(node, TensorInfo(outp_size, use_cnts.at(node), true))
          );
        } else {
          // No parameter aliasing, add a new entry to live_tensors and increase memory consumption
          curr_mem += outp_size;
          live_tensors.insert(
            std::make_pair(node, TensorInfo(outp_size, use_cnts.at(node), false))
          );
        }
      }
    }

    // Step 3: Check predecessors, add tensors that have zero use count to the free list
    /*
     * We always decrement the use count of the predecessor. Several cases for updating memory: 
     * 1. If the predecessor is a parameter or shares storage with a parameter, don't free it. 
     * 2. If the predecessor has expired (which means the op is an in-place op, otherwise something is wrong),
     *    then its memory is used by some other tensor, don't free memory. 
     * 3. Otherwise, add the predecessor to the free list if its use count drops to zero. 
     */
    for (auto pred : node->operands()) {
      const torch_lazy_tensors::ir::Node* pred_node = pred.node;
      LTC_CHECK(live_tensors.count(pred_node)) << "Predecessor " << pred_node->ToString() << " is not live!";
      auto& pred_node_info = live_tensors.at(pred_node);

      // Decrement remaining use count
      pred_node_info.use_cnt --;

      // Check for in-place op if the parameter has expired
      if (pred_node_info.is_expired) {
        LTC_CHECK(is_inplace) << "Op " << node->ToString() << ": operand " << pred_node->ToString()
          << " has expired but the op is not an in-place op!";
      }

      // Don't touch parameter memory, don't update memory for expired tensors
      if (!pred_node_info.is_param && !pred_node_info.is_expired && (pred_node_info.use_cnt == 0)) {
        to_be_freed.push_back(std::make_pair(pred_node, pred_node_info.size_mbs));
      }
    }

    // Step 4: Maintain peak memory
    peak_mem = (peak_mem > curr_mem) ? peak_mem : curr_mem;
    LTC_LOG(INFO) << "Current mem: " << curr_mem << "MBs";
  }
  LTC_LOG(INFO) << "Peak memory: " << peak_mem << "MBs";
  return peak_mem;
}
}