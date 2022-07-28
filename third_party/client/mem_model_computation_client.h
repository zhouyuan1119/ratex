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

namespace ratex {

using namespace lazy_tensors;

/*! \brief A set of PyTorch in-place ops. */
const std::unordered_set<std::string> pytorch_inplace_ops({
});
/*! \brief A set of PyTorch ops that only change the view of tensor but don't update tensors. */
const std::unordered_set<std::string> pytorch_view_changing_ops({
  "aten::permute",
  "aten::expand"
});

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
    MemModelData(std::string device, Shape shape)
        : BaseData(std::move(device), GetShapeData(std::move(shape))) {}

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
 private:
  /*! \brief Peak memory maintained by this client. */
  double peak_memory_ = 0.0;
};

lazy_tensors::ComputationClient* MemModelGet();

lazy_tensors::ComputationClient* MemModelGetIfInitialized();

/*! 
 * \brief Given a program specified in lazy tensor IR, traverse the graph and get the use count of
 * each node. 
 * \param nodes List of operators in the program, sorted in topological order 
 */
std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t> AnalyzeUseCount(
  const std::vector<const torch_lazy_tensors::ir::Node*>& topo_sorted_nodes);

/*!
 * \brief Collect the unique tensor IDs of a list of parameters. 
 * \param params The list of parameters in the program. 
 */
std::unordered_set<int64_t> GetParameterTensorIds(
  const std::vector<lazy_tensors::ComputationClient::DataPtr>& params);

/*!
 * \brief Collect the output tensor IDs at each lazy tensor IR nodes. The output map only contains
 * output tensors of the program because the input list of tensors only contains those tensors. 
 * \param tensors The list of output tensors of the program. 
 */
std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t> GetNodeTensorIdMap(
  const std::vector<torch_lazy_tensors::LazyTensor>& tensors);

/*!
 * \brief Calculate the size of the output of an op, in MBs, from the op's shape. 
 */
double CalculateMemFromShape(const lazy_tensors::client::ShapeData& shape);

/*! 
 * \brief Given a program specified in lazy tensor IR, traverse the graph and compute peak memory 
 * consumption in MBs. 
 * \param outputs Map from IR nodes to their output indices. For each (k, v) pair, node k is the v-th output. 
 * \param topo_sorted_nodes List of operators in the program, sorted in topological order 
 * \param params List of IR nodes that correspond to parameters
 * \param alias Mapping from parameters to outputs. For each (k, v) pair, the k-th output aliases with the v-th param. 
 * \param param_alias Mapping between parameters. For each(k, v) pair, the node k aliases with the v-th param in params. 
 * \param use_cnts Use count of each node, generated by AnalyzeUseCount()
 */
double CalculatePeakMem(const std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t>& outputs,
                        const std::vector<const torch_lazy_tensors::ir::Node*>& topo_sorted_nodes,
                        const std::vector<const torch_lazy_tensors::ir::Node*>& params,
                        const std::unordered_map<int64_t, int64_t>& alias,
                        const std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t>& param_alias,
                        const std::unordered_map<const torch_lazy_tensors::ir::Node*, int64_t>& use_cnts);

}
