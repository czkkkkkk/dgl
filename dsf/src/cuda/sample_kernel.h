/*!
 *  Copyright (c) 2022 by Contributors
 */

#ifndef DGL_DSF_CUDA_SAMPLE_KERNEL_H_
#define DGL_DSF_CUDA_SAMPLE_KERNEL_H_

#include "../utils.h"

namespace dgl {
namespace dsf {

struct SampleKernelOption {
  IdType *indptr, *indices;
  IdType* seeds;
  int64_t n_seeds, n_global_nodes, n_local_nodes;
  int64_t fanout;
  IdType *out_ptr, *out_indices;

  int nodes_per_group;
  int threads_per_peer;
};

}  // namespace dsf
}  // namespace dgl

#endif // DGL_DSF_CUDA_SAMPLE_KERNEL_H_