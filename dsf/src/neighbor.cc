/*!
 *  Copyright (c) 2022 by Contributors
 */
#include "neighbor.h"
#include "./cuda/sample_kernel.h"

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
IdArray SampleNeighbors(const SampleOption& option) {
  auto kernel_option = BuildKernelOption(option);
  IdArray ret = IdArray::Empty({option.seeds->shape[0] * option.fanout},
                               option.seeds->dtype, option.seeds->ctx);
  kernel_option.out_indices = ret.Ptr<IdType>();
  Sample(kernel_option);
  return ret;
}

}  // namespace dsf
}  // namespace dgl