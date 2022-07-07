/*!
 *  Copyright (c) 2022 by Contributors
 */

#ifndef DGL_DSF_CUDA_SAMPLE_KERNEL_H_
#define DGL_DSF_CUDA_SAMPLE_KERNEL_H_

#include "../context.h"
#include "../utils.h"

namespace dgl {
namespace dsf {

struct SampleKernelOption {
  IdType *indptr, *indices;
  IdType *global_nid_map;
  IdType *seeds;
  int64_t n_seeds, n_global_nodes, n_local_nodes;
  int64_t fanout;
  IdType *min_vids;
  int rank, world_size;
  IdType *out_rows, *out_cols;

  int nodes_per_block;
  int threads_per_peer;
};

void Sample(SampleKernelOption option);

}  // namespace dsf
}  // namespace dgl

#endif  // DGL_DSF_CUDA_SAMPLE_KERNEL_H_
