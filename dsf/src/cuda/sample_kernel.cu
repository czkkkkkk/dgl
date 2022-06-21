/*!
 *  Copyright (c) 2022 by Contributors
 */
#include "./sample_kernel.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "../conn/communicator.h"
#include "../utils.h"

#define DIVUP(x, y) ((x) + (y)-1) / (y)

#define INF 0x3f3f3f3f

namespace dgl {
namespace dsf {

// Sorting in a block
__device__ void BlockSort(IdType* in, IdType* index, int tid, int n) {
  typedef cub::BlockRadixSort<IdType, 128, 1, IdType> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;
  IdType k[1], v[1];
  if (tid >= n)
    k[0] = INF;
  else
    k[0] = in[tid];
  v[0] = tid;
  BlockRadixSort(temp_storage).Sort(k, v);
  if (tid < n) {
    in[tid] = k[0];
    index[tid] = v[0];
  }
}
__device__ void BlockCount(IdType* in, IdType* min_vids, int tid, int n,
                           int n_peers, IdType* offset) {
  if (tid <= n) offset[tid] = 0;
  __syncthreads();
  if (tid < n) {
    for (int peer = 0; peer < n_peers; ++peer) {
      if (min_vids[peer] <= in[tid] && in[tid] < min_vids[peer + 1]) {
        atomicAdd((unsigned long long*)(offset + peer + 1), 1);
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    for (int peer = 0; peer < n_peers; ++peer) {
      offset[peer + 1] += offset[peer];
    }
  }
}

__device__ void SendSeeds(IdType* input, IdType size, int tid, int n_threads,
                          VarArray output_array) {
  if (tid == 0) {
    *output_array.size = size;
  }
  IdType* output = (IdType*)output_array.data;
  while (tid < size) {
    output[tid] = input[tid];
    tid += n_threads;
  }
}

__global__ void FusedSampleKernel(SampleKernelOption option,
                                  Communicator* communicator) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int peer_id = tid / option.threads_per_peer;
  int local_tid = tid % option.threads_per_peer;
  int n_blocks = gridDim.x;
  int n_peers = option.world_size;
  int fanout = option.fanout;
  /*
  VarArray peer_seed_recv_buffer =
      communicator->block_conns[bid].conn_infos[peer_id].peer_buffer[0];
  VarArray peer_neigh_recv_buffer =
      communicator->block_conns[bid].conn_infos[peer_id].peer_buffer[1];
  VarArray my_seed_recv_buffer =
      communicator->block_conns[bid].conn_infos[peer_id].my_buffer[0];
  VarArray my_neigh_recv_buffer =
      communicator->block_conns[bid].conn_infos[peer_id].my_buffer[1];
  */

  __shared__ IdType sorted[1024], index[1024], send_sizes[MAX_CONN],
      send_offset[MAX_CONN + 1];

  int nodes_per_round = n_blocks * option.nodes_per_block;
  // Calculate # rounds
  int rounds = DIVUP(option.n_seeds, nodes_per_round);

  for (int round = 0; round < rounds; ++round) {
    // Find the range of seeds to sample

    int start_idx = round * nodes_per_round + bid * option.nodes_per_block;
    int end_idx = start_idx + option.nodes_per_block;
    if (start_idx > option.n_seeds) start_idx = option.n_seeds;
    if (end_idx > option.n_seeds) end_idx = option.n_seeds;
    IdType* start = option.seeds + start_idx;
    IdType* end = option.seeds + end_idx;
    int size = end_idx - start_idx;

    // size <= blockDim.x
    if (tid < size) {
      sorted[tid] = start[tid];
      index[tid] = tid;
    }
    __syncthreads();
    BlockSort(sorted, index, tid, size);
    BlockCount(sorted, option.min_vids, tid, size, n_peers, send_offset);
    if (tid < size) {
      printf("[Rank %d], tid %d, sorted %lld, index %lld\n", option.rank, tid,
             sorted[tid], index[tid]);
    }
    if (tid <= option.world_size) {
      printf("[Rank %d], tid %d, offset %lld\n", option.rank, tid,
             send_offset[tid]);
    }

    // IdType peer_start = send_offset[peer_id];
    // IdType peer_end = send_offset[peer_id + 1];
    // Conduct partition in a block
    //   Send seeds to a buffer, where is on the CommArgs
    // SendSeeds(sorted + peer_start, peer_start - peer_end, local_tid,
    //        option.threads_per_peer, peer_seed_recv_buffer);

    // Global to local nid
    // LocalSample(my_seed_recv_buffer, peer_neigh_recv_buffer);
    // CopyNeighToOutptr(my_neigh_recv_buffer, option.out_ptr,
    // option.out_idices);
  }
}

void Sample(SampleKernelOption option) {
  LOG(ERROR) << "Try to sample...";
  auto* communicator = Context::Global()->communicator.get();
  int n_threads = 128;
  int n_blocks = communicator->n_blocks;
  option.nodes_per_block = n_threads;
  option.threads_per_peer = n_threads / option.world_size;
  FusedSampleKernel<<<n_blocks, n_threads>>>(option,
                                             communicator->dev_communicator);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaStreamSynchronize(0));
  LOG(ERROR) << "Finished sample...";
}

}  // namespace dsf
}  // namespace dgl