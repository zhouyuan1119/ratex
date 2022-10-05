/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "client/mem_model_computation_client.h"

#include <fstream>
#include <iostream>

#include "ratex/csrc/compiler/utils.h"
#include "ratex/csrc/value_ext/value.h"
#include "ratex/csrc/pass_ext/pass.h"
#include "ratex/csrc/utils/file.h"
#include "env_vars.h"

#include "lazy_tensors/computation_client/nnc_computation_client.h"
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ops/dummy.h"

namespace ratex {

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
  auto* computation = static_cast<GenericComputationMemModel*>(instance.computation.get());

  auto post_order_nodes = computation->GetPostOrderNodes();
  auto alias = computation->GetAlias();
  auto outputs = computation->GetOutputs();
  LTC_LOG(INFO) << "Outputs: ";
  std::unordered_map<const Node*, int64_t> outputs_map;
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

  // ReplaceUseWithDummy(post_order_nodes);

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

  // Add some information to the output node info
  AnnotateLayerName();
  AnnotateNodeType(params, outputs);
  ConvertForOutput();

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

OutputMap<int64_t> MemModelComputationClient::AnalyzeUseCount(
  const std::vector<const Node*>& topo_sorted_nodes) {
  OutputMap<int64_t>  use_cnts;
  for (auto* node : topo_sorted_nodes) {
    for (auto pred : node->operands()) {
      if (use_cnts.count(pred))
        use_cnts[pred] += 1;
      else
        use_cnts[pred] = 1;
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

std::vector<double> MemModelComputationClient::CalculateMemFromShape(const lazy_tensors::Shape& shape) {
  std::vector<double> sizes;
  if (shape.tuple_shapes_size() == 0) {
    // Single tensor, non-tuple
    int64_t size = GetElementSizeInBytes(shape.element_type());
    for (int64_t dim : shape.dimensions()) {
      size *= dim;
    }
    sizes.push_back(size / kMegaBytes);
  } else {
    // Tuple
    for (auto elem_shape : shape.tuple_shapes()) {
      int64_t size = GetElementSizeInBytes(elem_shape.element_type());
      for (int64_t dim : elem_shape.dimensions()) {
        size *= dim;
      }
      sizes.push_back(size / kMegaBytes);
    }
  }
  
  return sizes;
}

/*! 
 * \brief Use a heuristic to check if an op is an in-place op. 
 * \param node Pointer to the IR node (operator). 
 * \param live_tensors A map from live tensors to their corresponding TensorInfos. 
 */
bool IsInplaceOp(const Node* node, const OutputMap<TensorInfo>& live_tensors) {  
  /* 
   * Notice that due to the limitation of lazy tensor IR, we cannot find in-place ops from the 
   * op() method of IR nodes. As a result, we take the following heuristic:
   * - If the op's first operand has zero remaining use count after this op, and 
   * - All of the views of this operand has zero remaining use count, and
   * - The first output of this op has the same shape as the operand
   * Then we treat this op as an in-place op. We only check the first-input vs. the first-output
   * for now. Usually this should suffice. 
   */

  // Special handling of the dummy op
  // if (node->op() == *ops::ltc_dummy)
  //   return true;

  // Filter out nodes with no inputs (e.g., constant nodes)
  if (node->operands().size() < 1)
    return false;
  
  // Check the first operand's use count
  auto pred_node = node->operand(0);
  auto pred_node_info = live_tensors.at(pred_node);
  if (pred_node_info.use_cnt > 1)
    return false;

  // Check if the first operand is a view
  if (pred_node_info.is_view) {
    // If the first operand is a view, must check its original tensor
    auto viewed_tensor = pred_node_info.viewing;
    // The original tensor must have no active views other than this one
    // It must also have no future uses
    auto viewed_tensor_info = live_tensors.at(viewed_tensor);
    if ((viewed_tensor_info.use_cnt > 0) || (viewed_tensor_info.viewers.size() > 1) ||
        (viewed_tensor_info.size_mbs != pred_node_info.size_mbs))
      return false;
    // If this check passes, go on to check shape
  }
  
  // Check output shape
  auto pred_node_shape = pred_node.shape();
  auto output_shape = node->shape();
  if (output_shape.tuple_shapes_size() > 0)
    output_shape = output_shape.tuple_shapes(0);
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

double MemModelComputationClient::CalculatePeakMem(const std::unordered_map<const Node*, int64_t>& outputs_map,
                        const std::vector<const Node*>& topo_sorted_nodes,
                        const std::vector<const Node*>& params,
                        const std::unordered_map<int64_t, int64_t>& alias,
                        const std::unordered_map<const Node*, int64_t>& param_alias,
                        const OutputMap<int64_t>& use_cnts) {
  double curr_mem = 0.0;
  
  /* Data structures for tracking memory in each layer */

  // Memory tracker
  memory_breakdown_.clear();
  LayerMemInfo curr_layer_info;
  // Set of output tensors (activations)
  OutputSet outputs_in_layer; 
  // Input parameters and activation sizes
  OutputMap<double> layer_params;
  OutputMap<double> layer_input_act;

  // Maintain the current set of live tensors
  OutputMap<TensorInfo> live_tensors;
  // A list of tensors that have reached the end of their lifetime, together with their sizes
  std::vector<std::pair<const Output, double>> to_be_freed;

  // For dumping node information
  node_info_.clear();
  node_info_for_dumping_.clear();

  // Parameters persist in the memory
  for (auto param_node : params) {
    // We assume parameters are always non-tuples
    Output param_tensor(param_node, 0);
    // Don't include useless parameters
    if (use_cnts.count(param_tensor)) {
      double param_mem = CalculateMemFromShape(Shape(param_tensor.shape()))[0];
      curr_mem += param_mem;
      // Insert parameters into the live set, they are never removed
      live_tensors.insert(
        std::make_pair(
          param_tensor, 
          TensorInfo(param_mem, use_cnts.at(param_tensor), true))
      );
    }
  }
  LTC_LOG(INFO) << "Param total: " << curr_mem << " MBs";
  // Also add the aliased parameters, but here we don't increment memory
  for (auto param_with_alias : param_alias) {
    auto param_node = param_with_alias.first;
    Output param_tensor(param_node, 0);
    if (use_cnts.count(param_tensor)) {
      auto aliased_param_id = param_with_alias.second;
      auto& aliased_param_node = params.at(aliased_param_id);
      Output aliased_param_tensor(aliased_param_node, 0);
      auto& aliased_param_info = live_tensors.at(aliased_param_tensor);
      live_tensors.insert(
        std::make_pair(
          param_tensor, 
          TensorInfo(aliased_param_info.size_mbs, use_cnts.at(param_tensor), true)
        )
      );
    }
  }

  double peak_mem = curr_mem;
  double layer_peak_mem = curr_mem;

  // Assuming all nodes are sorted in topological order and the ops will be executed exactly in this order
  for (auto* node : topo_sorted_nodes) {
    LTC_LOG(INFO) << "|" << node->ToString() << ", uses: " << node->uses().size();
    // Step 0: add the node to our list
    node_info_.push_back(NodeInfo(node));

    // Step 1: Purge any tensors that can be freed
    /* 
     * A live tensor can be safely freed if:
     * 1. It has a use count of zero;
     * 2. It is not a parameter or aliasing with parameters;
     * 3. It has not expired, otherwise its memory is taken over by another tensor and we free that
     *    tensor later instead. 
     * 4. All of its viewers have been deleted. 
     */
    for (auto tensors_with_size : to_be_freed) {
      auto t = tensors_with_size.first;
      auto size_mbs = tensors_with_size.second;
      LTC_CHECK(live_tensors.count(t)) << "Tensor " << t.ToString() << " is not in live set!";
      auto& node_info = live_tensors.at(t);
      LTC_CHECK(node_info.use_cnt <= 0) << "Tensor " << t.ToString() << " with use count of " << node_info.use_cnt << " is freed!";
      LTC_CHECK(!node_info.is_param) << "Parameter tensor " << t.ToString() << " is freed!";
      LTC_CHECK(!node_info.is_expired) << "Expired tensor " << t.ToString() << " is freed!";
      LTC_CHECK(node_info.viewers.size() == 0) << "Tensor " << t.ToString() << " has " << node_info.viewers.size() << " viewers but is freed!";
      curr_mem -= (node_info.is_view) ? 0.0 : size_mbs;
      live_tensors.erase(t);
      LTC_LOG(INFO) << "|-Erase " << t.ToString() << " for " << ((node_info.is_view) ? 0.0 : size_mbs) << " MBs memory";
    }
    to_be_freed.clear();

    // LTC_LOG(INFO) << "Dump live tensors:"; 
    // for (auto tensor_with_info : live_tensors) {
    //   auto t = tensor_with_info.first;
    //   auto info = tensor_with_info.second;
    //   LTC_LOG(INFO) << "|Live tensor: " << t.ToString();
    //   LTC_LOG(INFO) << "|" << info.dump();
    // }

    // Step 1.1: Go through the predecessors of each node for analyzing per-layer parameters and inputs
    for (auto operand : node->operands()) {
      LTC_CHECK(live_tensors.count(operand)) << "Predecessor " << operand.ToString() << " is not live!";
      // Ignore outputs generated in this layer
      if (outputs_in_layer.count(operand))
        continue;
      auto operand_info = live_tensors.at(operand);
      LTC_CHECK(!operand_info.is_expired) << "Predecessor " << operand.ToString() << " has expired!";
      // If the operand is a view, use the actual tensor instead
      auto actual_tensor = operand;
      double actual_size = operand_info.size_mbs;
      if (operand_info.is_view) {
        actual_tensor = operand_info.viewing;
        LTC_CHECK(live_tensors.count(actual_tensor)) << "Tensor being viewed " << actual_tensor.ToString() << " is not live!";
        actual_size = live_tensors.at(actual_tensor).size_mbs;
      }
      // Insert these operands into their corresponding sets to avoid counting for multiple times
      if (operand_info.is_param) {
        layer_params.insert(std::make_pair(actual_tensor, actual_size));
      } else {
        layer_input_act.insert(std::make_pair(actual_tensor, actual_size));
      }
    }
    
    // Step 2: Add the output of the current op to the live set and update current memory
    auto outp_sizes = CalculateMemFromShape(node->shape());
    // LTC_CHECK(use_cnts.count(node)) << "Node " << node->ToString() << " does not have use count!";

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
    if (node->op() != *ops::ltc_device_data) {
      if (IsViewChangingOp(node->op().op)) {
        LTC_LOG(INFO) << "|-View-changing op";
        LTC_CHECK(node->operands().size() == 1) << "View-changing ops with more than one inputs are currently not supported!";
        LTC_CHECK(node->num_outputs() == 1) << "View-changing ops with more than one outputs are currently not supported!";
        auto pred_tensor = node->operand(0);
        LTC_CHECK(live_tensors.count(pred_tensor)) << "Predecessor " << pred_tensor.ToString() << " is not live!";
        auto& pred_node_info = live_tensors.at(pred_tensor);
        // Put a new entry
        /*
         * 1. We inherent the is_param field from the predecessor. 
         * 2. Handle view-sharing differently based on whether the predecesor is a view or not. 
         */
        auto viewing_tensor = (pred_node_info.is_view) ? pred_node_info.viewing : pred_tensor;
        auto outp_tensor = Output(node, 0);
        const int64_t use_cnt = use_cnts.count(outp_tensor) ? use_cnts.at(outp_tensor) : 0;
        live_tensors.insert(
          std::make_pair(
            outp_tensor, 
            TensorInfo(outp_sizes[0], use_cnt, pred_node_info.is_param, viewing_tensor)
          )
        );
        live_tensors.at(viewing_tensor).viewers.insert(outp_tensor);
        outputs_in_layer.insert(outp_tensor);
      } else if (IsInplaceOp(node, live_tensors)) {
        LTC_LOG(INFO) << "|-Inplace op";
        is_inplace = true;
        auto pred_tensor = node->operand(0);
        // Mark the entry of the input as expired
        LTC_CHECK(live_tensors.count(pred_tensor)) << "Predecessor " << pred_tensor.ToString() << " is not live!";
        auto& pred_node_info = live_tensors.at(pred_tensor);
        pred_node_info.is_expired = true;
        // If the predecesor is a view, mark its source as expired too
        if (pred_node_info.is_view) {
          auto& viewing_tensor_info = live_tensors.at(pred_node_info.viewing);
          viewing_tensor_info.is_expired = true;
        }

        // Put a new entry for each output, special handling for the first one and assume the other outputs
        // are all newly allocated tensors
        // If the predecessor is a parameter or aliases with a parameter, then the new tensor shares 
        // memory with a parameter. The new tensor cannot be viewing anything. 
        for (int i = 0; i < node->num_outputs(); i ++) {
          auto outp = Output(node, i);
          bool outp_is_param = (i == 0) ? pred_node_info.is_param : false;
          const int64_t use_cnt = use_cnts.count(outp) ? use_cnts.at(outp) : 0;
          live_tensors.insert(
            std::make_pair( 
              outp, 
              TensorInfo(outp_sizes[i], use_cnt, outp_is_param)
            )
          );
          outputs_in_layer.insert(outp);
          curr_mem += (i == 0) ? 0 : outp_sizes[i];
        }
      } else {
        // Not in-place op or view-changing op, check for I/O param aliasing
        if (outputs_map.count(node) && alias.count(outputs_map.at(node))) {
          is_alias = true;
          // If there is I/O param aliasing, this is the output and mark the param as expired
          auto param_node = params.at(alias.at(outputs_map.at(node)));
          auto param_tensor = Output(param_node, 0);
          LTC_LOG(INFO) << "|-Aliases with param " << param_node->ToString();
          LTC_CHECK(live_tensors.count(param_tensor)) << "Parameter " << param_node->ToString() << " is not live!";
          LTC_CHECK(node->num_outputs() == 1) << "Node " << node->ToString() << " aliases with param " << param_node->ToString()
            << ", but has more than one outputs!";
          auto& param_node_info = live_tensors.at(param_tensor);
          param_node_info.is_expired = true;
          LTC_CHECK(outp_sizes[0] == param_node_info.size_mbs) << "I/O aliasing but tensor sizes mismatch: "
            << outp_sizes[0] << " vs. " << param_node_info.size_mbs;
          // Put a new entry
          // In this case the new tensor cannot be viewing any other tensor
          auto outp_tensor = Output(node, 0);
          const int64_t use_cnt = use_cnts.count(outp_tensor) ? use_cnts.at(outp_tensor) : 0;
          live_tensors.insert(
            std::make_pair(outp_tensor, TensorInfo(outp_sizes[0], use_cnt, true))
          );
          outputs_in_layer.insert(outp_tensor);
        } else {
          // No parameter aliasing, add a new entry to live_tensors and increase memory consumption
          for (int i = 0; i < node->num_outputs(); i ++) {
            auto outp = Output(node, i);
            const int64_t use_cnt = use_cnts.count(outp) ? use_cnts.at(outp) : 0;
            live_tensors.insert(
              std::make_pair( 
                outp, 
                TensorInfo(outp_sizes[i], use_cnt, false)
              )
            );
            outputs_in_layer.insert(outp);
            curr_mem += outp_sizes[i];
          }
        }
      }
    }

    // Step 3: Maintain memory information. The output size of this op has been added, and all tensors that
    // are no longer useful before this op have been freed at this point. 
    peak_mem = std::max(peak_mem, curr_mem);
    layer_peak_mem = std::max(layer_peak_mem, curr_mem);
    LTC_LOG(INFO) << "|-Current mem: " << curr_mem << "MBs";

    // Step 3.1: At layer boundaries, update the memory information of the whole layer over here
    auto metadata = dynamic_cast<LayerBoundaryMetaData*>(node->user_metadata());
    if (metadata != nullptr || (node->op() == *ops::ltc_dummy)) {
      std::string layer_name = "";
      if (metadata != nullptr)
        layer_name = metadata->name;
      else {
        layer_name = NodeCast<ops::Dummy>(node, ops::ltc_dummy)->name();
      }
      // Peak memory consumption in this layer
      curr_layer_info.peak_mem_mbs = layer_peak_mem;
      // Output tensors in this layer
      for (auto t : outputs_in_layer) {
        // Only consider tensors that are still live, not expired, not param, not view
        if (live_tensors.count(t)) {
          auto t_info = live_tensors.at(t);
          if (!t_info.is_param && !t_info.is_expired && !t_info.is_view) {
            curr_layer_info.outp_act_mbs += t_info.size_mbs;
            LTC_LOG(INFO) << "Counting output act " << t.ToString() << " for " << t_info.size_mbs << " MBs";
          }
        }
      }
      // Input tensors, including parameters and activations
      for (auto p : layer_params) {
        curr_layer_info.param_mbs += p.second;
        LTC_LOG(INFO) << "Counting param " << p.first.ToString() << " for " << p.second << " MBs";
      }
      for (auto p : layer_input_act) {
        curr_layer_info.inp_mbs += p.second;
        LTC_LOG(INFO) << "Counting input act " << p.first.ToString() << " for " << p.second << " MBs";
      }
      // "Isolated peak memory", this is basically the peak memory inside this layer, minus the size
      // of live tensors that are untouched in this layer. "Untouched" means:
      // 1. It is not generated in this layer (i.e., not in outputs_in_layers)
      // 2. It is not used by this layer (i.e., not in layer_params or layer_input_act)
      // 3. For in-place updates and views: any tensor that is in-place updated or viewed inside the 
      //    layer will appear in either layer_params or layer_input_act, thus are already considered
      // 4. If the tensor is expired or is a view, we don't decrement its size
      curr_layer_info.peak_mem_isolated_mbs = layer_peak_mem;
      for (auto p : live_tensors) {
        auto output = p.first;
        auto t_info = p.second;
        if (!outputs_in_layer.count(output) && !layer_input_act.count(output) && 
            !layer_params.count(output) && !t_info.is_expired && !t_info.is_view) {
          curr_layer_info.peak_mem_isolated_mbs -= p.second.size_mbs;
          LTC_LOG(INFO) << "Purge tensor " << output.ToString() << " because it is untouched!";
        }
      }
      // LTC_LOG(INFO) << "mem_at_layer_begin_mbs: " << mem_at_layer_begin_mbs;
      // curr_layer_info.peak_mem_isolated_mbs = layer_peak_mem - 
      //   (mem_at_layer_begin_mbs - curr_layer_info.inp_mbs - curr_layer_info.param_mbs);
      LTC_LOG(INFO) << "input_mbs: " << curr_layer_info.inp_mbs;
      LTC_LOG(INFO) << "param_mbs: " << curr_layer_info.param_mbs;
      LTC_LOG(INFO) << "output_mbs: " << curr_layer_info.outp_act_mbs;
      LTC_LOG(INFO) << "peak_mem: " << layer_peak_mem;
      LTC_LOG(INFO) << "peak_mem_isolated_mbs: " << curr_layer_info.peak_mem_isolated_mbs;
      outputs_in_layer.clear();
      layer_params.clear();
      layer_input_act.clear();
      // Create a new entry
      memory_breakdown_.insert(std::make_pair(layer_name, curr_layer_info));
      curr_layer_info = LayerMemInfo();
      layer_peak_mem = 0.0;
    }

    // LTC_LOG(INFO) << "Dump live tensors:"; 
    // for (auto tensor_with_info : live_tensors) {
    //   auto t = tensor_with_info.first;
    //   auto info = tensor_with_info.second;
    //   LTC_LOG(INFO) << "|Live tensor: " << t.ToString();
    //   LTC_LOG(INFO) << "|" << info.dump();
    // }
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
      LTC_CHECK(live_tensors.count(pred)) << "Predecessor " << pred.ToString() << " is not live!";
      auto& pred_node_info = live_tensors.at(pred);

      // Decrement remaining use count
      pred_node_info.use_cnt --;

      // Check for in-place op if the parameter has expired
      if (pred_node_info.is_expired) {
        LTC_CHECK(is_inplace || (is_alias && (pred.node == params.at(alias.at(outputs_map.at(node))))) ) 
          << "Op " << node->ToString() << ": operand " << pred.ToString()
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
        to_be_freed.push_back(std::make_pair(pred, pred_node_info.size_mbs));
        if (pred_node_info.is_view) {
          auto& viewing_node_info = live_tensors.at(pred_node_info.viewing);
          viewing_node_info.viewers.erase(pred);
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

void MemModelComputationClient::AnnotateLayerName() {
  int64_t total_nodes = node_info_.size();
  int64_t start = 0, curr = 0;

  while (curr < total_nodes) {
    auto curr_node = node_info_[curr].node;
    auto metadata = dynamic_cast<LayerBoundaryMetaData*>(curr_node->user_metadata());
    if (metadata != nullptr || (curr_node->op() == *ops::ltc_dummy)) {
      std::string layer_name = "";
      if (metadata != nullptr)
        layer_name = metadata->name;
      else {
        layer_name = NodeCast<ops::Dummy>(curr_node, ops::ltc_dummy)->name();
      }
      for (int64_t i = start; i <= curr; i ++) {
        node_info_[i].SetName(layer_name);
      }
      start = curr + 1;
    }
    curr ++;
  }

  // Annotate the optimizer as well
  for (int64_t i = start; i < total_nodes; i ++) {
    node_info_[i].SetName("optimizer");
  }
}

/*! \brief Find the gradient node from a multiplication op */
const Node* FindGrad(const Node* mul_node) {
  LTC_CHECK(mul_node->op().op == at::aten::mul) << "Expecting mul op, but got " << mul_node->ToString();
  auto op0 = mul_node->operand(0).node;
  auto op1 = mul_node->operand(1).node;
  LTC_CHECK((op0->op().op == at::aten::expand) ^ (op1->op().op == at::aten::expand)) << "Exactly one operand of the mul must be expanding a constant (learning rate)!";
  if (op0->op().op == at::aten::expand)
    return op1;
  else
    return op0;
}

void MemModelComputationClient::AnnotateNodeType(const std::vector<const Node*>& params, 
                                                 const std::vector<const Node*>& outputs) {
  std::unordered_set<const Node*> params_set(params.begin(), params.end());
  std::unordered_set<const Node*> output_set(outputs.begin(), outputs.end());
  std::unordered_set<const Node*> grads;
  std::unordered_set<const Node*> wgts;
  for (auto& i : node_info_) {
    auto node = i.node;
    if (node->op() == *ops::ltc_device_data) {
      // Mark all device data nodes as inputs at the beginning, we will change it later if it is weight
      i.SetAsInput();
    } else {
      // Similarly, mark everything else as activation at first, we will change it later if it is grad
      i.SetAsAct();
    }

    /* 
     * We find gradient tensors by looking at optimizer updates. Optimizer updates that actually writes
     * to the weight tensors can be located. The op is usually an add, and the two operators are the
     * weight tensor and the gradient times learning rate (a constant). We can identify all weight
     * and gradient tensors in this way. A by-product of this is that all device_data nodes that are
     * not touched by optimizer updates remain as input tensors. 
     */
    if ((i.layer_name == "optimizer") && (output_set.count(node)) && (node->op().op == at::aten::add)) {
      // LTC_CHECK(node->op().op == at::aten::add) << "Expecting add op at optimizer outputs, but got " << node->ToString();
      auto op0 = node->operand(0).node;
      auto op1 = node->operand(1).node;
      LTC_CHECK(params_set.count(op0) ^ params_set.count(op1)) << "Exactly one operand of the add must be parameter!";
      if (params_set.count(op0)) {
        wgts.insert(op0);
        grads.insert(FindGrad(op1));
      } else {
        wgts.insert(op1);
        grads.insert(FindGrad(op0));
      }
    }
  }

  for (auto& i : node_info_) {
    if (grads.count(i.node))
      i.SetAsGrad();
    else if (wgts.count(i.node))
      i.SetAsWeight();
  }
}

void MemModelComputationClient::ConvertForOutput() {
  for (auto info : node_info_) {
    std::string node_ty_str;
    switch (info.node_type) {
      case INPUT: { node_ty_str = "input";  break; }
      case WGT: { node_ty_str = "weight"; break; }
      case ACT: { node_ty_str = "act"; break; }
      case GRAD: { node_ty_str = "grad"; break; }
      default: { node_ty_str = "unknown"; break; }
    }
    NodeInfoForOutput info_for_output(
      info.node->ToString(), info.node->op().ToString(), 
      info.node->id(), info.layer_name, node_ty_str);
    for (auto use : info.node->uses()) {
      info_for_output.AddUse(use.node->id());
    }
    for (auto operand : info.node->operands()) {
      info_for_output.AddOperand(operand.node->id(), operand.index);
    }
    auto& shape = info.node->shape();
    if (shape.IsTuple()) {
      for (int i = 0; i < info.node->shape().tuple_shapes_size(); i ++) {
        auto tuple_field_shape = shape.tuple_shapes(i);
        info_for_output.AddOutputShape(
          PrimitiveTypeName(tuple_field_shape.element_type()),
          std::vector<int64_t>(tuple_field_shape.dimensions().begin(), tuple_field_shape.dimensions().end())
        );
      }
    } else {
      info_for_output.AddOutputShape(
          PrimitiveTypeName(shape.element_type()),
          std::vector<int64_t>(shape.dimensions().begin(), shape.dimensions().end())
      );
    }
    node_info_for_dumping_.push_back(info_for_output);
  }
}

}
