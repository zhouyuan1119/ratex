/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_map>

#include "client/base_computation_client.h"
#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/client_data.h"
#include "lazy_tensor_core/csrc/tensor.h"
#include "ratex/csrc/compiler/mem_model_lowering_context.h"

namespace ratex {

using namespace lazy_tensors;
using namespace torch_lazy_tensors::compiler;
using namespace torch_lazy_tensors::compiler::mem_model_lowering_backend;
using namespace torch_lazy_tensors::ir;

const double kMegaBytes = 1048576.0;

/*! \brief A set of PyTorch in-place ops. */
const std::unordered_set<std::string> pytorch_inplace_ops({
});
/*! \brief A set of PyTorch ops that only change the view of tensor but don't update tensors. */
const std::unordered_set<std::string> pytorch_view_changing_ops({
  "aten::permute",
  "aten::expand",
  "aten::view",
  "aten::select" // for torch.narrow and torch.select
});

/*! \brief Struct for storing live tensor information in memory analysis. */
struct TensorInfo {
  // Construct tensor info for a view
  TensorInfo(double size, int64_t uses, bool param, const Output orig_tensor) 
  : size_mbs(size), use_cnt(uses), is_param(param), viewing(orig_tensor) { is_view = true; }

  // Construct tensor info for a normal tensor
  TensorInfo(double size, int64_t uses, bool param) 
  : size_mbs(size), use_cnt(uses), is_param(param) {}

  std::string dump() {
    std::stringstream ss;
    ss << "size: " << size_mbs
       << ", use_cnt: " << use_cnt
       << ", is_param: " << is_param
       << ", is_expired: " << is_expired;
    if (is_view)
      ss << ", viewing: " << viewing.ToString();
    return ss.str();
  }

  // Size of tensor in MBs
  double size_mbs;
  // Use count of this tensor
  int64_t use_cnt;
  // True if the tensor is a parameter or shares storage with a parameter
  bool is_param;
  // True if the tensor has undergone an in-place update and is replaced by a newer version
  bool is_expired = false;
  // If this "tensor" is actually a view, keep a pointer to the original tensor that keeps memory
  bool is_view = false;
  const Output viewing;
  // If this tensor has multiple views, keep a set of pointers to each view
  OutputSet viewers = {};
};

enum NodeType {
  INPUT = 0,
  WGT,
  ACT,
  GRAD,
  UNKNOWN
};

/*! \brief Struct for dumping node information. */
struct NodeInfo {

  NodeInfo(const torch_lazy_tensors::ir::Node* n) : node(n) {}
  void SetName(const std::string& n) { layer_name = n; }
  void SetAsInput() { node_type = INPUT; }
  void SetAsWeight() { node_type = WGT; }
  void SetAsAct() { node_type = ACT; }
  void SetAsGrad() { node_type = GRAD; }

  const torch_lazy_tensors::ir::Node* node;
  std::string layer_name = "";
  NodeType node_type = UNKNOWN;
};

/*!
 * \brief This class defines the computation client for memory modeling. It only
 * examines the lazy tensor IR and generates the memory model. It does not do
 * any lowering, nor does it actually executes code on the device. 
 */
class MemModelComputationClient : public BaseComputationClient {
 public:

  // Must overload BaseData because it is an abstract class
  struct MemModelData : public BaseData {
   public:
    MemModelData(std::string device, Shape shape, bool is_param=false)
        : BaseData(std::move(device), GetShapeData(std::move(shape)), is_param) {}

    /*! 
     * \brief Handle is just an integer. Use with care, may cause issues. 
     */ 
    OpaqueHandle GetOpaqueHandle() override {
      return reinterpret_cast<OpaqueHandle>(this);
    }

    /*!
     * \brief Assigning any other real data to this fake data does nothing. 
     */
    void Assign(const Data& data) override {}

    /*!
     * \brief We should never access this data, because it never has real value in it. 
     */
    bool HasValue() const override {
      return true;
    }
   
  };

  struct MemModelComputation : public BaseComputation {
    MemModelComputation(
        std::shared_ptr<GenericComputation> computation, 
        ProgramShape program_shape,
        std::vector<std::string> devices,
        const std::unordered_map<int64_t, int64_t>& alias = {})
        : BaseComputation(computation, program_shape, devices, alias) {
      peak_memory_mbs = 0.0;
    }

    MemModelComputation(
        std::shared_ptr<GenericComputation> computation, 
        ProgramShape program_shape,
        std::vector<std::string> devices, 
        double peak_mem,
        const std::unordered_map<int64_t, int64_t>& alias = {})
        : BaseComputation(computation, program_shape, devices, alias),
          peak_memory_mbs(peak_mem) {
    }

    // Peak memory of this computation, in MBs
    double peak_memory_mbs;
  };
 

  // For all other methods, we temporarily just duplicate the function signatures
  // of whatever RAFComputationClient has. Will remove the redundant parts later. 

  MemModelComputationClient(Options options) : BaseComputationClient(options) {}

  /*! \brief Create a computation client for memory modeling. */
  static std::unique_ptr<ComputationClient> Create();

  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  /*! \brief This function wraps lazy tensors into BaseData. It does not allocate space. */
  std::vector<DataPtr> TransferToServer(lazy_tensors::Span<const TensorSource> tensors) override;
  
  /*! \brief Converts BaseData into Literals. No actual work done and no space allocated. */
  std::vector<Literal> TransferFromServer(lazy_tensors::Span<const DataPtr> handles) override;

  /*! 
   * \brief Analyzes the graph and builds the mem model. This function is supposed to run for the
   * initial graph (whole model) as well as every query from the scheduler (subset of layers). As a 
   * first version we just let it look at the whole graph and compute peak memory usage. We might 
   * consider some sort of caching later. 
   */
  ComputationPtr Compile(CompileInstance instances) override;

  /*! 
   * \brief This is just a dummy function to satisfy the interface requirements. It returns an empty
   * list of tensors. We'll have a separate API to get the peak memory. 
   */
  std::vector<DataPtr> ExecuteComputation(const Computation& computation,
                                          lazy_tensors::Span<const DataPtr> arguments,
                                          const std::string& device,
                                          const ExecuteComputationOptions& options) override;
  
  /*! \brief Interface function to get the peak memory */
  virtual double GetPeakMemory() override { return peak_memory_; }
  virtual std::unordered_map<std::string, LayerMemInfo> GetMemoryBreakDown() override { return memory_breakdown_; }

  /*! \brief Interface function to get node information. */
  virtual std::vector<NodeInfoForOutput> GetNodeInfo() { return node_info_for_dumping_; }

 private:
  /*! \brief Peak memory maintained by this client. */
  double peak_memory_ = 0.0;
  /*! \brief Memory breakdown of each layer. */
  std::unordered_map<std::string, LayerMemInfo> memory_breakdown_;
  /*! \brief Node information for internal processing. */
  std::vector<NodeInfo> node_info_;
  /*! \brief Node information for dumping. */
  std::vector<NodeInfoForOutput> node_info_for_dumping_;

  /*! 
   * \brief Given a program specified in lazy tensor IR, traverse the graph and get the use count of
   * each tensor. 
   * \param nodes List of operators in the program, sorted in topological order 
   * \return A map from ir::Output to int64_t, where each (k, v) pair means tensor k has a use count of
   * v. Notice that we use ir::Output instead of ir::Node. ir::Output roughly represents a tensor in the
   * program, where the op that generated the tensor (node) and the index of the tensor (index) are both
   * recorded. The latter is useful if the tensor is part of a tuple. Tensors that have zero use count
   * are not in the map. 
   */
  OutputMap<int64_t> AnalyzeUseCount(
    const std::vector<const torch_lazy_tensors::ir::Node*>& topo_sorted_nodes);

  /*!
   * \brief Calculate the size of the output of an op, in MBs, from the op's shape. If the op generates
   * a tuple, the returned vector will have multiple elements. 
   */
  std::vector<double> CalculateMemFromShape(const lazy_tensors::Shape& shape);

  /*! 
   * \brief Given a program specified in lazy tensor IR, traverse the graph and compute peak memory 
   * consumption in MBs. This function also provides a layer-wise breakdown of memory consumption. 
   * \param outputs Map from IR nodes to their output indices. For each (k, v) pair, node k is the v-th output. 
   * \param topo_sorted_nodes List of operators in the program, sorted in topological order 
   * \param params List of IR nodes that correspond to parameters
   * \param alias Mapping from parameters to outputs. For each (k, v) pair, the k-th output aliases with the v-th param. 
   * \param param_alias Mapping between parameters. For each(k, v) pair, the node k aliases with the v-th param in params. 
   * \param use_cnts Use count of each tensor, generated by AnalyzeUseCount()
   */
  double CalculatePeakMem(const std::unordered_map<const Node*, int64_t>& outputs,
                          const std::vector<const Node*>& topo_sorted_nodes,
                          const std::vector<const Node*>& params,
                          const std::unordered_map<int64_t, int64_t>& alias,
                          const std::unordered_map<const Node*, int64_t>& param_alias,
                          const OutputMap<int64_t>& use_cnts);

  /*! \brief Annotate the layer name to node info. */
  void AnnotateLayerName();

  /*! \brief Annotate the node type info. */
  void AnnotateNodeType(const std::vector<const Node*>& params, const std::vector<const Node*>& outputs);

  /*! \brief Convert the node info to output format. */
  void ConvertForOutput();
};

lazy_tensors::ComputationClient* MemModelGet();

lazy_tensors::ComputationClient* MemModelGetIfInitialized();

}
