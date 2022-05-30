/*!
 *  Copyright (c) 2022 by Contributors
 */
#include "../utils.h"
#include "./sample_kernel.h"

namespace dgl {
namespace dsf {

__global__ void FusedSampleKernel(const SampleKernelOption& option) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int peer_id = tid / option.threads_per_peer;
  int local_tid = tid % option.threads_per_peer;

  __shared__ IdType sorted[1024], index[1024], send_sizes[MAX_CONN],
      send_offset[MAX_CONN + 1];

  // Calculate # rounds
  int round = 10;

  for (int i = 0; i < round; ++i) {
    // Find the range of seeds to sample
    int64_t *start, *end;

    // Conduct partition in a block
    //   Send seeds to a buffer, where is on the CommArgs
    IdType* seed_recv_buffer; // Unique buffer for one block and one conn;
    int recv_size = CopySeeds(sorted, index, send_offset, seed_recv_buffer);
    // LocalSample(); Write the output directly to the neighbor recv buffer. 
    IdType* neighbor_recv_buffer;
    // Copy the results on the neighbor_recv_buffer back to the out_indices;
  }
}

}  // namespace dsf
}  // namespace dgl