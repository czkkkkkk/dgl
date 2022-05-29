/*!
 *  Copyright (c) 2022 by Contributors
 */

#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>

#include "./context.h"
#include "./utils.h"
#include "c_api_common.h"
#include "dmlc/logging.h"

using namespace dgl::runtime;

namespace dgl {
namespace dsf {

void InitCoordinator(Context* context) {
  int master_port = GetEnvParam("MASTER_PORT", 12608);
  context->coor = std::unique_ptr<Coordinator>(
      new Coordinator(context->rank, context->world_size, master_port));
}

void Initialize(int rank, int world_size) {
  LOG(INFO) << "[Rank " << rank << "] Initialize DSF";
  auto* context = Context::Global();
  context->rank = rank;
  context->world_size = world_size;
  InitCoordinator(context);

  LOG(INFO) << "[Rank " << rank << "] Finish initialization";
}

DGL_REGISTER_GLOBAL("dsf._CAPI_DGLDSFInitialize")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      int rank = args[0];
      int world_size = args[1];
      Initialize(rank, world_size);
    });

}  // namespace dsf
}  // namespace dgl
