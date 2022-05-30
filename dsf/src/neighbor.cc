/*!
 *  Copyright (c) 2022 by Contributors
 */
#include "neighbor.h"
#include "./cuda/sample_kernel.h"

namespace dgl {
namespace dsf {

IdArray SampleNeighbors(SampleOption option) {
  auto csr_mat = option.hg->GetCSRMatrix(0);
  SampleKernelOption kernel_option;
  kernel_option.indptr = csr_mat.indptr.Ptr<IdType>();
  kernel_option.indices = csr_mat.indices.Ptr<IdType>();
  kernel_option.seeds = option.seeds.Ptr<IdType>();
  kernel_option.n_seeds = option.seeds->shape[0];
  kernel_option.n_local_nodes = option.n_local_nodes;
  kernel_option.n_global_nodes = option.n_global_nodes;
  kernel_option.fanout = option.fanout;
  IdArray ret = IdArray::Empty({option.seeds->shape[0] * option.fanout}, option.seeds->dtype, option.seeds->ctx);
  kernel_option.out_indices = ret.Ptr<IdType>();
}

}
}  // namespace dgl