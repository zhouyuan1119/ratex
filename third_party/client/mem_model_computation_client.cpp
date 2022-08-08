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
  // LTC_LOG(INFO) << "In MemModelComputationClient::Compile";
  auto* computation = static_cast<GenericComputationMemModel*>(instance.computation.get());
  // LTC_LOG(INFO) << "Got computation!";

  auto post_order_nodes = computation->GetPostOrderNodes();
  auto alias = computation->GetAlias();
  auto outputs = computation->GetOutputs();
  LTC_LOG(INFO) << "Outputs: ";
  std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t> outputs_map;
  for (int64_t i = 0; i < outputs.size(); i ++) {
    outputs_map.insert(std::make_pair(outputs[i], i));
    LTC_LOG(INFO) << "|-" << outputs[i]->ToString();
  }
  auto params = computation->GetParameters();
  auto param_alias = computation->GetParamAlias();
  LTC_LOG(INFO) << "I/O tensor alias:";
  for (auto p : alias) {
    auto outp_idx = p.first;
    auto param_idx = p.second;
    LTC_LOG(INFO) << "|-Output " << outputs[outp_idx]->ToString() << " <-> " << " param " << params[param_idx]->ToString();
  }
  LTC_LOG(INFO) << "Param alias: ";
  for (auto p : param_alias) {
    auto node = p.first;
    auto second_idx = p.second;
    LTC_LOG(INFO) << "|-Param " << node->ToString() << " <->  param " << params[second_idx]->ToString();
  }
  // LTC_LOG(INFO) << "Got info!";

  // Walk the graph and get the use count of each node. 
  // We cannot leverage the use count in lazy tensor IR because over there the
  // uses are maintained in a set, which will cause issues for our analysis. 
  auto use_cnts = AnalyzeUseCount(post_order_nodes);
  // LTC_LOG(INFO) << "Got use counts!";

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
                                         param_alias,
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
    case PrimitiveType::PRED:
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

bool IsInplaceOp(const torch_lazy_tensors::ir::Node* node, 
                 const std::unordered_map<const torch_lazy_tensors::ir::Node*, TensorInfo>& live_tensors) {  
  // Filter out nodes with no inputs (e.g., constant nodes)
  if (node->operands().size() < 1)
    return false;
  
  // Currently don't handle multi-output nodes
  if (node->num_outputs() > 1)
    return false;

  // Check the first operand's use count
  auto pred_node = node->operand(0).node;
  auto pred_node_info = live_tensors.at(pred_node);
  if (pred_node_info.use_cnt > 1)
    return false;

  // Check if the first operand is a view
  if (pred_node_info.viewing != nullptr)
    return false;
  
  // Check output shape
  auto pred_node_shape = pred_node->shape();
  auto output_shape = node->shape();
  if (!(pred_node_shape == output_shape))
    return false;

  // Check for views of the predecessor
  for (auto view : pred_node_info.viewers) {
    auto view_info = live_tensors.at(view);
    if (view_info.use_cnt > 0)
      return false;
  }

  return true;
}

bool IsViewChangingOp(const c10::Symbol op) {
  return pytorch_view_changing_ops.count(std::string(op.toQualString()));
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
                        const std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t>& param_alias,
                        const std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t>& use_cnts) {
  double curr_mem = 0.0;

  // Maintain the current set of live tensors
  std::unordered_map<const torch_lazy_tensors::ir::Node*, TensorInfo> live_tensors;
  // A list of tensors that have reached the end of their lifetime, together with their sizes
  std::vector<std::pair<const torch_lazy_tensors::ir::Node*, double>> to_be_freed;

  // Parameters persist in the memory
  for (auto param : params) {
    // Only include parameters that are used in the graph to avoid counting parameters from the previous batch
    if (use_cnts.count(param) && (use_cnts.at(param) > 0)) {
      double param_mem = CalculateMemFromShape(Shape(param->shape()));
      curr_mem += param_mem;
      // Insert parameters into the live set, they are never removed
      live_tensors.insert(std::make_pair(param, TensorInfo(param_mem, use_cnts.at(param), true, nullptr)));
    }
  }
  LTC_LOG(INFO) << "Param total: " << curr_mem << " MBs";
  // Also add the aliased parameters, but here we don't increment memory
  for (auto param_with_alias : param_alias) {
    auto param = param_with_alias.first;
    auto aliased_param_id = param_with_alias.second;
    LTC_CHECK(live_tensors.count(param) == 0) << "Parameter " << param->ToString() 
      << " is aliased with another parameter but it's already inserted into live tensors!";
    auto& aliased_param_info = live_tensors.at(params.at(aliased_param_id));
    live_tensors.insert(std::make_pair(param, TensorInfo(aliased_param_info.size_mbs, use_cnts.at(param), true, nullptr)));
  }

  double peak_mem = curr_mem;

  // Assuming all nodes are sorted in topological order and the ops will be executed exactly in this order
  for (auto* node : topo_sorted_nodes) {
    LTC_LOG(INFO) << "|" << node->ToString() << ", uses: " << node->uses().size();
    // Step 1: Purge any tensors that can be freed
    /* 
     * A live tensor can be safely freed if:
     * 1. It has a use count of zero;
     * 2. It is not a parameter or aliasing with parameters;
     * 3. It has not expired, otherwise its memory is taken over by another tensor and we free that
     *    tensor later instead. 
     * 4. All of its viewers have been deleted. 
     */
    for (auto node_with_size : to_be_freed) {
      auto node_ptr = node_with_size.first;
      auto size_mbs = node_with_size.second;
      auto& node_info = live_tensors.at(node_ptr);
      LTC_CHECK(node_info.use_cnt <= 0) << "Node " << node_ptr->ToString() << " with use count of " << node_info.use_cnt << " is freed!";
      LTC_CHECK(!node_info.is_param) << "Parameter node " << node_ptr->ToString() << " is freed!";
      LTC_CHECK(!node_info.is_expired) << "Expired node " << node_ptr->ToString() << " is freed!";
      LTC_CHECK(node_info.viewers.size() == 0) << "Node " << node_ptr->ToString() << " has " << node_info.viewers.size() << " viewers but is freed!";
      curr_mem -= (node_info.viewing) ? 0.0 : size_mbs;
      live_tensors.erase(node_ptr);
      LTC_LOG(INFO) << "|-Erase " << node_ptr->ToString() << " for " << ((node_info.viewing) ? 0.0 : size_mbs) << " MBs memory";
    }
    to_be_freed.clear();
    // Step 2: Add the output of the current op to the live set and update current memory
    double outp_size = CalculateMemFromShape(node->shape());
    LTC_CHECK(use_cnts.count(node)) << "Node " << node->ToString() << " does not have use count!";

    /* 
     * There are several cases here:
     * 1. This node is device_data(), which means the memory is already added when processing parameters. 
     *    In this case we don't do anything. 
     * 2. This node is not device_data(), but the op is an in-place op. In this case we don't increment 
     *    memory, but make the tensor associated with the input "expired" and create a new entry in 
     *    the live tensor set to represent the output. 
     * 3. This node is not device_data(), not an in-place op, but a "view-changing" op that changes 
     *    the view of a tensor without modifying its data (e.g., permute). In this case we do the same
     *    as above, except that the input tensor is not marked as expired. 
     * 4. This node is not device_data() nor an in-place/view-changing op, but the output of the node 
     *    shares memory with a parameter. This is defined by the alias map. In this case we don't increment 
     *    memory, but update the TensorInfo associated with the parameter to reflect the use count of this 
     *    tensor.  
     * 5. This node is not device_data(), not an in-place/view-changin op, and the output does not share 
     *    memory with a parameter. In this case we add the size of the output to curr_mem, and create 
     *    a new entry in live_tensors for this output. 
     */
    bool is_inplace = false;
    bool is_alias = false;
    if (node->op() != *torch_lazy_tensors::ir::ops::ltc_device_data) {
      // Check for in-place op. 
      /* 
       * Notice that due to the limitation of lazy tensor IR, we cannot find in-place ops from the 
       * op() method of IR nodes. As a result, we take the following heuristic:
       * - If the op's first operand is not a view, and 
       * - It has zero remaining use count after this op, and 
       * - All of the views of this operand has zero remaining use count, and
       * - The output of this op has the same shape as the operand
       * Then we treat this op as an in-place op. 
       */
      if (IsInplaceOp(node, live_tensors)) {
        LTC_LOG(INFO) << "|-Inplace op";
        is_inplace = true;
        auto pred_node = node->operands()[0].node;
        // Mark the entry of the input as expired
        LTC_CHECK(live_tensors.count(pred_node)) << "Predecessor " << pred_node->ToString() << " is not live!";
        auto& pred_node_info = live_tensors.at(pred_node);
        pred_node_info.is_expired = true;
        // Put a new entry. 
        // If the predecessor is a parameter or aliases with a parameter, then the new tensor shares 
        // memory with a parameter. 
        live_tensors.insert(
          std::make_pair(
            node, 
            TensorInfo(outp_size, use_cnts.at(node), pred_node_info.is_param, pred_node_info.viewing)
          )
        );
      } else if (IsViewChangingOp(node->op().op)) {
        LTC_LOG(INFO) << "|-View-changing op";
        LTC_CHECK(node->operands().size() == 1) << "View-changing ops with more than one inputs are currently not supported!";
        auto pred_node = node->operands()[0].node;
        LTC_CHECK(live_tensors.count(pred_node)) << "Predecessor " << pred_node->ToString() << " is not live!";
        auto& pred_node_info = live_tensors.at(pred_node);
        // Put a new entry
        /*
         * 1. Similarly, we inherent the is_param field from the predecessor. 
         * 2. Handle view-sharing differently based on whether the predecesor is a view or not. 
         */
        auto viewing_node = (pred_node_info.viewing) ? pred_node_info.viewing : pred_node;
        live_tensors.insert(
          std::make_pair(
            node, 
            TensorInfo(outp_size, use_cnts.at(node), pred_node_info.is_param, viewing_node)
          )
        );
        live_tensors.at(viewing_node).viewers.insert(node);
      } else {
        // Not in-place op or view-changing op, check for I/O param aliasing
        if (outputs_map.count(node) && alias.count(outputs_map.at(node))) {
          is_alias = true;
          // If there is I/O param aliasing, this is the output and mark the param as expired
          auto param_node = params.at(alias.at(outputs_map.at(node)));
          LTC_LOG(INFO) << "|-Aliases with param " << param_node->ToString();
          LTC_CHECK(live_tensors.count(param_node)) << "Parameter " << param_node->ToString() << " is not live!";
          auto& param_node_info = live_tensors.at(param_node);
          param_node_info.is_expired = true;
          LTC_CHECK(outp_size == param_node_info.size_mbs) << "I/O aliasing but tensor sizes mismatch: "
            << outp_size << " vs. " << param_node_info.size_mbs;
          // Put a new entry
          // In this case the new tensor cannot be viewing any other tensor
          live_tensors.insert(
            std::make_pair(node, TensorInfo(outp_size, use_cnts.at(node), true, nullptr))
          );
        } else {
          // No parameter aliasing, add a new entry to live_tensors and increase memory consumption
          curr_mem += outp_size;
          // Similarly, since the op is not a view-changing op, the new tensor is not a view of an
          // existing tensor
          live_tensors.insert(
            std::make_pair(node, TensorInfo(outp_size, use_cnts.at(node), false, nullptr))
          );
        }
      }
    }

    // Step 3: Maintain peak memory. The output size of this op has been added, and all tensors that
    // are no longer useful before this op have been freed at this point. 
    peak_mem = (peak_mem > curr_mem) ? peak_mem : curr_mem;
    LTC_LOG(INFO) << "|-Current mem: " << curr_mem << "MBs";

    // Step 4: Check predecessors
    /*
     * Notice that we only "free" views that are no longer used here. We free for another round at 
     * step one (when processing the next op) to free the tensor that actually holds memory if all 
     * of its views have reached the end of their life times. Non-view predecessors are also freed 
     * over there.  
     */
    // Potential optimization: hold a list of tensors to be freed so that we don't need to traverse
    // the whole live set every time
    for (auto pred : node->operands()) {
      const torch_lazy_tensors::ir::Node* pred_node = pred.node;
      LTC_CHECK(live_tensors.count(pred_node)) << "Predecessor " << pred_node->ToString() << " is not live!";
      auto& pred_node_info = live_tensors.at(pred_node);

      // Decrement remaining use count
      pred_node_info.use_cnt --;

      // Check for in-place op if the parameter has expired
      if (pred_node_info.is_expired) {
        LTC_CHECK(is_inplace || (is_alias && (pred_node == params.at(alias.at(outputs_map.at(node))))) ) 
          << "Op " << node->ToString() << ": operand " << pred_node->ToString()
          << " has expired! This is only allowed when (1) the op is an in-place op, or (2) the op's output"
          << " aliases with this predecessor. ";
      }

      // Erase from the live tensor set if the predecessor is no longer useful. Memory consumption 
      // is updated at step 1 of the next iteration
      /*
       * 1. Not parameter or alias with parameter
       * 2. Have not been in-place updated
       * 3. Has no use count
       * 4. Is not currently viewed by some other tensor
       */
      if (!pred_node_info.is_param && !pred_node_info.is_expired && (pred_node_info.use_cnt == 0) && 
          (pred_node_info.viewers.size() == 0)) {
        to_be_freed.push_back(std::make_pair(pred_node, pred_node_info.size_mbs));
        if (pred_node_info.viewing) {
          auto& viewing_node_info = live_tensors.at(pred_node_info.viewing);
          viewing_node_info.viewers.erase(pred_node);
          if ((!viewing_node_info.is_param) && (!viewing_node_info.is_expired) && 
              (viewing_node_info.use_cnt == 0) && (viewing_node_info.viewers.size() == 0)) {
            to_be_freed.push_back(std::make_pair(pred_node_info.viewing, viewing_node_info.size_mbs));
          }
        }
      }
    }
  }
  LTC_LOG(INFO) << "Peak memory: " << peak_mem << "MBs";
  return peak_mem;
}
}