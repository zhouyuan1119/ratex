/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/compiler/base_backend_impl.h"
#include "client/mem_model_computation_client.h"

namespace torch_lazy_tensors {
namespace compiler {

/*! \brief Creates an interface for getting the mem model computation client */
class MemModelBackendImpl : public BaseBackendImpl {
 public:
  lazy_tensors::ComputationClient* GetComputationClient() const override {
    return ratex::MemModelGet();
  }

  lazy_tensors::ComputationClient* GetComputationClientIfInitialized() const override {
    return ratex::MemModelGetIfInitialized();
  }
};

/*! \brief Register our client */
BackendImplRegistry* mem_model_backend_impl_registry =
    GetBackendImplRegistry()->AddBackendImpl(new MemModelBackendImpl(), 20);

BackendRegistrar g_registrar(GetBackendImplRegistry()->GetBackendImpl());

}  // namespace compiler
}  // namespace torch_lazy_tensors
