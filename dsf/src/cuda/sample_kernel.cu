/*!
 *  Copyright (c) 2022 by Contributors
 */
#include "./sample_kernel.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

#include "../conn/communicator.h"
#include "../utils.h"
#include "./comm_sync.cuh"

#define DIVUP(x, y) ((x) + (y)-1) / (y)
#define INF 0x3f3f3f3f

namespace dgl {
namespace dsf {

template <int BLOCK_SIZE>
__device__ IdType GatherMaxRounds(IdType rounds, int local_tid,
                                  IdType* peer_recv, IdType* my_recv,
                                  CommSync* sync) {
  sync->Unset();
  sync->PreComm();
  if (local_tid == 0) {
    *peer_recv = rounds;
  }
  sync->PostComm();
  IdType peer_rounds;
  if (local_tid == 0) {
    peer_rounds = *my_recv;
  } else {
    peer_rounds = 1;
  }
  __syncthreads();

  typedef cub::BlockReduce<IdType, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ IdType shared_max_rounds;
  IdType max_rounds = BlockReduce(temp_storage).Reduce(peer_rounds, cub::Max());
  if (local_tid == 0) {
    shared_max_rounds = max_rounds;
  }
  __syncthreads();
  return shared_max_rounds;
}

// Sorting in a block
template <int BLOCK_SIZE>
__device__ void BlockSort(IdType* in, IdType* index, int tid, int n) {
  typedef cub::BlockRadixSort<IdType, BLOCK_SIZE, 1, IdType> BlockRadixSort;
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

template <int WORLD_SIZE>
__device__ void BlockCount(IdType* in, IdType* min_vids, int tid, int n,
                           IdType* offset) {
  if (tid <= WORLD_SIZE) offset[tid] = 0;
  __syncthreads();
  if (tid < n) {
    for (int peer = 0; peer < WORLD_SIZE; ++peer) {
      if (min_vids[peer] <= in[tid] && in[tid] < min_vids[peer + 1]) {
        atomicAdd((unsigned long long*)(offset + peer + 1), 1);
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    for (int peer = 0; peer < WORLD_SIZE; ++peer) {
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

__device__ void LocalSample(VarArray seeds, VarArray output, IdType seed_offset,
                            int tid, int n_threads, int fanout, IdType* indptr,
                            IdType* indices) {
  const uint64_t random_seed = 7777777;
  curandState rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);
  int64_t n_seeds = *seeds.size;
  if (tid == 0) {
    *output.size = n_seeds * fanout;
  }
  IdType* seeds_ptr = (IdType*)seeds.data;
  IdType* outptr = (IdType*)output.data;
  int group_size = 1;
  while (group_size < fanout && group_size * 2 <= n_threads) {
    group_size *= 2;
  }
  int n_groups = n_threads / group_size;
  int row = tid / group_size;
  int col = tid % group_size;
  while (row < n_seeds) {
    IdType local_nid = seeds_ptr[row] - seed_offset;
    IdType in_row_start = indptr[local_nid];
    IdType out_row_start = row * fanout;
    IdType deg = indptr[local_nid + 1] - in_row_start;
    for (int idx = col; idx < fanout; idx += group_size) {
      // sequentially sample seeds
      // const int64_t edge = idx % deg;
      const int64_t edge = curand(&rng) % deg;
      outptr[out_row_start + idx] = indices[in_row_start + edge];
    }
    row += n_groups;
  }
}

__device__ void CopyNeighToOutptr(VarArray neighbors, IdType n_seeds,
                                  IdType* sorted_seeds, IdType* index,
                                  int fanout, int tid, int n_threads,
                                  IdType* out_rows, IdType* out_cols) {
  IdType* neigh_ptr = (IdType*)neighbors.data;
  while (tid < n_seeds * fanout) {
    int seed_ptr = tid / fanout;
    int offset = tid % fanout;
    IdType idx = index[seed_ptr];
    out_rows[idx * fanout + offset] = neigh_ptr[tid];
    out_cols[idx * fanout + offset] = sorted_seeds[seed_ptr];
    tid += n_threads;
  }
}

#define SWITCH_BLOCK_SIZE(val, BLOCK_SIZE, ...) \
  do {                                          \
    if ((val) == 64) {                          \
      constexpr int BLOCK_SIZE = 64;            \
      { __VA_ARGS__ }                           \
    } else if ((val) == 128) {                  \
      constexpr int BLOCK_SIZE = 128;           \
      { __VA_ARGS__ }                           \
    } else if ((val) == 256) {                  \
      constexpr int BLOCK_SIZE = 256;           \
      { __VA_ARGS__ }                           \
    } else if ((val) == 512) {                  \
      constexpr int BLOCK_SIZE = 512;           \
      { __VA_ARGS__ }                           \
    } else {                                    \
      CHECK(false);                             \
    }                                           \
  } while (0)

#define SWITCH_WORLD_SIZE(val, WORLD_SIZE, ...) \
  do {                                          \
    if ((val) == 1) {                           \
      constexpr int WORLD_SIZE = 1;             \
      { __VA_ARGS__ }                           \
    } else if ((val) == 2) {                    \
      constexpr int WORLD_SIZE = 2;             \
      { __VA_ARGS__ }                           \
    } else if ((val) == 4) {                    \
      constexpr int WORLD_SIZE = 4;             \
      { __VA_ARGS__ }                           \
    } else if ((val) == 8) {                    \
      constexpr int WORLD_SIZE = 8;             \
      { __VA_ARGS__ }                           \
    } else {                                    \
      CHECK(false);                             \
    }                                           \
  } while (0)

template <int BLOCK_SIZE, int WORLD_SIZE>
__global__ void FusedSampleKernel(SampleKernelOption option,
                                  Communicator* communicator) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int rank = option.rank;
  int peer_id = tid / option.threads_per_peer;
  int local_tid = tid % option.threads_per_peer;
  int n_blocks = gridDim.x;
  int n_peers = option.world_size;
  int fanout = option.fanout;
  ConnInfo* conn_info = communicator->block_conns[bid].conn_infos + peer_id;
  VarArray peer_seed_recv_buffer = conn_info->peer_buffer[0];
  VarArray peer_neigh_recv_buffer = conn_info->peer_buffer[1];
  VarArray my_seed_recv_buffer = conn_info->my_buffer[0];
  VarArray my_neigh_recv_buffer = conn_info->my_buffer[1];

  CommSync sync(conn_info->my_ready, conn_info->my_done, conn_info->peer_ready,
                conn_info->peer_done, local_tid == 0);

  __shared__ IdType sorted[BLOCK_SIZE], index[BLOCK_SIZE],
      send_offset[WORLD_SIZE + 1];

  int nodes_per_round = n_blocks * option.nodes_per_block;
  // Calculate # rounds
  int rounds = DIVUP(option.n_seeds, nodes_per_round);
  rounds = GatherMaxRounds<BLOCK_SIZE>((IdType)rounds, local_tid,
                                       peer_seed_recv_buffer.size,
                                       my_seed_recv_buffer.size, &sync);

  for (int round = 0; round < rounds; ++round) {
    // Find the range of seeds to sample
    int start_idx = round * nodes_per_round + bid * option.nodes_per_block;
    int end_idx = start_idx + option.nodes_per_block;
    if (start_idx > option.n_seeds) start_idx = option.n_seeds;
    if (end_idx > option.n_seeds) end_idx = option.n_seeds;
    IdType* start = option.seeds + start_idx;
    int size = end_idx - start_idx;

    IdType* out_rows = option.out_rows + start_idx * fanout;
    IdType* out_cols = option.out_cols + start_idx * fanout;

    // size <= blockDim.x
    if (tid < size) {
      sorted[tid] = start[tid];
      index[tid] = tid;
    }
    __syncthreads();
    BlockSort<BLOCK_SIZE>(sorted, index, tid, size);
    BlockCount<WORLD_SIZE>(sorted, option.min_vids, tid, size, send_offset);
    __syncthreads();

    sync.Unset();

    IdType peer_start = send_offset[peer_id];
    IdType peer_end = send_offset[peer_id + 1];
    IdType send_size = peer_end - peer_start;
    // Conduct partition in a block
    //   Send seeds to a buffer, where is on the CommArgs
    sync.PreComm();
    SendSeeds(sorted + peer_start, send_size, local_tid,
              option.threads_per_peer, peer_seed_recv_buffer);
    sync.PostComm();

    sync.PreComm();
    LocalSample(my_seed_recv_buffer, peer_neigh_recv_buffer,
                option.min_vids[rank], local_tid, option.threads_per_peer,
                fanout, option.indptr, option.indices);
    sync.PostComm();

    CopyNeighToOutptr(my_neigh_recv_buffer, send_size, sorted,
                      index + send_offset[peer_id], fanout, local_tid,
                      option.threads_per_peer, out_rows, out_cols);

    __syncthreads();
  }
}

void Sample(SampleKernelOption option) {
  auto* communicator = Context::Global()->communicator.get();
  int n_threads = GetEnvParam("BLOCK_SIZE", 128);
  int n_blocks = communicator->n_blocks;
  option.nodes_per_block = n_threads;
  option.threads_per_peer = n_threads / option.world_size;

  SWITCH_BLOCK_SIZE(n_threads, BLOCK_SIZE, {
    SWITCH_WORLD_SIZE(option.world_size, WORLD_SIZE, {
      FusedSampleKernel<BLOCK_SIZE, WORLD_SIZE>
          <<<n_blocks, n_threads>>>(option, communicator->dev_communicator);
    });
  });
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaStreamSynchronize(0));
}

}  // namespace dsf
}  // namespace dgl