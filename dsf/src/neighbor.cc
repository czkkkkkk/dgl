/*!
 *  Copyright (c) 2022 by Contributors
 */
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>
#include <chrono>
#include <thread>

#include "./cuda/sample_kernel.h"
#include "neighbor.h"

using namespace dgl::runtime;

namespace dgl {
namespace dsf {

SampleKernelOption BuildKernelOption(const SampleOption& option) {
  SampleKernelOption kernel_option;
  auto csr_mat = option.hg->GetCSRMatrix(0);
  kernel_option.indptr = csr_mat.indptr.Ptr<IdType>();
  kernel_option.indices = csr_mat.indices.Ptr<IdType>();
  kernel_option.global_nid_map = option.global_nid_map.Ptr<IdType>();
  kernel_option.seeds = option.seeds.Ptr<IdType>();
  kernel_option.n_seeds = option.seeds->shape[0];
  kernel_option.n_local_nodes = option.n_local_nodes;
  kernel_option.n_global_nodes = option.n_global_nodes;
  kernel_option.fanout = option.fanout;
  kernel_option.min_vids = option.min_vids.Ptr<IdType>();
  kernel_option.rank = Context::Global()->rank;
  kernel_option.world_size = Context::Global()->world_size;
  return kernel_option;
}

HeteroGraphPtr SampleNeighbors(const SampleOption& option) {
  auto kernel_option = BuildKernelOption(option);
  IdArray out_rows = IdArray::Empty({option.seeds->shape[0] * option.fanout},
                                    option.seeds->dtype, option.seeds->ctx);
  IdArray out_cols = IdArray::Empty({option.seeds->shape[0] * option.fanout},
                                    option.seeds->dtype, option.seeds->ctx);
  kernel_option.out_rows = out_rows.Ptr<IdType>();
  kernel_option.out_cols = out_cols.Ptr<IdType>();
  Sample(kernel_option);
  auto ret = UnitGraph::CreateFromCOO(
      1, option.n_global_nodes, option.n_global_nodes, out_rows, out_cols);
  return ret;
}

// FIXME: Gather and broadbast the number of seeds takes a lot of time
int ComputeMaxNSeeds(IdType size) {
  auto* coor = Context::Global()->coor.get();
  auto sizes = coor->Gather(size);
  IdType max_size = size;
  for (auto v : sizes) {
    max_size = std::max(max_size, v);
  }
  max_size = coor->Broadcast(max_size);
  return max_size;
}

DGL_REGISTER_GLOBAL("dsf.neighbor._CAPI_DGLDSFSampleNeighbors")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef hg = args[0];
      IdType n_global_nodes = args[1];
      IdType n_local_nodes = args[2];
      IdArray min_vids = args[3];
      IdArray seeds = args[4];
      IdArray global_nid_map = args[5];
      int fanout = args[6];

      SampleOption option;
      option.hg = hg;
      option.seeds = seeds;
      option.global_nid_map = global_nid_map;
      option.n_local_nodes = n_local_nodes;
      option.n_global_nodes = n_global_nodes;
      option.fanout = fanout;
      option.min_vids = min_vids;
      auto ret = SampleNeighbors(option);
      *rv = HeteroGraphRef(ret);
    });

}  // namespace dsf
}  // namespace dgl