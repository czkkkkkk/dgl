/*!
 *  Copyright (c) 2022 by Contributors
 */

#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>

#include "./context.h"
#include "./utils.h"
#include "./core.h"
#include "c_api_common.h"
#include "conn/communicator.h"
#include "conn/connection.h"
#include "dmlc/logging.h"
#include "conn/nvmlwrap.h"

using namespace dgl::runtime;

namespace dgl {
namespace dsf {

void InitCoordinator(Context* context) {
  int master_port = GetEnvParam("MASTER_PORT", 12608);
  context->coor = std::unique_ptr<Coordinator>(
      new Coordinator(context->rank, context->world_size, master_port));
}


void InitCommunicator(Context* context) {
  wrapNvmlInit();
  context->communicator = std::unique_ptr<Communicator>(new Communicator());
  SetupCommunicator(context->communicator.get());
}

void Initialize(int rank, int world_size) {
  LOG(INFO) << "[Rank " << rank << "] Initializing DSF context";
  auto* context = Context::Global();
  context->rank = rank;
  context->world_size = world_size;
  CUDACHECK(cudaSetDevice(rank));
  InitCoordinator(context);
  InitCommunicator(context);

  LOG(INFO) << "[Rank " << rank << "] Finished initialization";
}

DGL_REGISTER_GLOBAL("dsf._CAPI_DGLDSFInitialize")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      int rank = args[0];
      int world_size = args[1];
      Initialize(rank, world_size);
    });

}  // namespace dsf
}  // namespace dgl
