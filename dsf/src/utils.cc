/*!
 *  Copyright (c) 2022 by Contributors
 */
#include "utils.h"

#include <arpa/inet.h>
#include <dgl/array.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>
#include <dmlc/logging.h>
#include <netdb.h>
#include <netinet/ip.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstring>

#include "./context.h"
#include "graph/unit_graph.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {
namespace dsf {

int GetAvailablePort() {
  struct sockaddr_in addr;
  addr.sin_port = htons(0);   // 0 means let system pick up an available port.
  addr.sin_family = AF_INET;  // IPV4
  addr.sin_addr.s_addr = htonl(INADDR_ANY);  // set addr to any interface

  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (0 != bind(sock, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))) {
    DLOG(WARNING) << "bind()";
    return 0;
  }
  socklen_t addr_len = sizeof(struct sockaddr_in);
  if (0 != getsockname(sock, (struct sockaddr*)&addr, &addr_len)) {
    DLOG(WARNING) << "getsockname()";
    return 0;
  }

  int ret = ntohs(addr.sin_port);
  close(sock);
  return ret;
}

std::string GetHostName() {
  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);

  struct addrinfo hints = {0};
  hints.ai_family = AF_UNSPEC;
  hints.ai_flags = AI_CANONNAME;

  struct addrinfo* res = 0;
  std::string fqdn;
  if (getaddrinfo(hostname, 0, &hints, &res) == 0) {
    // The hostname was successfully resolved.
    fqdn = std::string(res->ai_canonname);
    freeaddrinfo(res);
  } else {
    // Not resolved, so fall back to hostname returned by OS.
    LOG(FATAL) << " ERROR: No HostName.";
  }
  return fqdn;
}

IdArray ToGlobal(IdArray nids, IdArray global_nid_map) {
  CHECK(nids->ctx.device_type == kDLCPU);
  CHECK(global_nid_map->ctx.device_type == kDLCPU);
  IdType length = nids->shape[0];
  IdArray ret = IdArray::Empty({length}, nids->dtype, nids->ctx);
  IdType* ret_ptr = ret.Ptr<IdType>();
  IdType* nids_ptr = nids.Ptr<IdType>();
  IdType* global_nid_map_ptr = global_nid_map.Ptr<IdType>();
  for (int i = 0; i < length; ++i) {
    ret_ptr[i] = global_nid_map_ptr[nids_ptr[i]];
  }
  return ret;
}

IdArray RebalanceRandom(IdArray ids, int batch_size) {
  auto* coor = Context::Global()->coor.get();
  auto ids_vec = ids.ToVector<int64_t>();
  auto vecs = coor->Gather(ids_vec);
  if (coor->IsRoot()) {
    int total = 0;
    for (const auto& vec : vecs) {
      total += vec.size();
    }
    int world_size = coor->GetWorldSize();
    auto flatten = Flatten(vecs);
    std::random_shuffle(flatten.begin(), flatten.end());
    int size_per_rank = total / world_size;
    flatten.resize(size_per_rank * world_size);
    for (int i = 0; i < world_size; ++i) {
      vecs[i] = std::vector<int64_t>(flatten.begin() + i * size_per_rank,
                                     flatten.begin() + (i + 1) * size_per_rank);
    }
  }
  auto ret = coor->Scatter(vecs);
  return NDArray::FromVector(ret);
}

/**
 * @brief Rebalance local node ids of all ranks so that each rank have
 * the same number of node ids. It may drop some node ids to keep balance.
 * Note that the output are global node ids.
 *
 * @param nids local node ids
 * @param batch_size pack as much ids as possible according to the batch_size
 * @param global_nid_map
 *
 * @return balanced global node ids
 */
DGL_REGISTER_GLOBAL("dsf.utils._CAPI_DGLDSFRebalanceNIds")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      IdArray nids = args[0];
      int batch_size = args[1];
      IdArray global_nid_map = args[2];
      std::string mode = args[3];
      IdArray global_nids = ToGlobal(nids, global_nid_map);
      if (mode == "random") {
        auto ret = RebalanceRandom(global_nids, batch_size);
        *rv = ret;
      } else {
        LOG(FATAL) << "Unknown rebalance mode: " << mode;
      }
    });

void ToGlobalInplace(IdArray nids, IdArray global_nid_map) {
  CHECK(nids->ctx.device_type == kDLCPU);
  CHECK(global_nid_map->ctx.device_type == kDLCPU);
  IdType length = nids->shape[0];
  IdArray ret = IdArray::Empty({length}, nids->dtype, nids->ctx);
  IdType* ret_ptr = ret.Ptr<IdType>();
  IdType* nids_ptr = nids.Ptr<IdType>();
  IdType* global_nid_map_ptr = global_nid_map.Ptr<IdType>();
  for (int i = 0; i < length; ++i) {
    nids_ptr[i] = global_nid_map_ptr[nids_ptr[i]];
  }
}

DGL_REGISTER_GLOBAL("dsf.utils._CAPI_DGLDSFCSRToGlobalId")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef hg = args[0];
      IdArray global_nid_map = args[1];
      assert(hg->NumEdgeTypes() == 1);
      dgl_type_t etype = 0;
      CSRMatrix csr_mat = hg->GetCSRMatrix(etype);
      ToGlobalInplace(csr_mat.indices, global_nid_map);
    });

}  // namespace dsf
}  // namespace dgl
