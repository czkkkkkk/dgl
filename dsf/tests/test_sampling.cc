/*!
 *  Copyright (c) 2022 by Contributors
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "neighbor.h"
#include "graph/unit_graph.h"
#include "dlpack/dlpack.h"

using namespace dgl;
using namespace dgl::runtime;
using namespace dgl::dsf;

template<typename T>
void ExpVectorEq(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  EXPECT_EQ(lhs.size(), rhs.size());
  for(size_t i = 0; i < lhs.size(); ++i) {
    EXPECT_EQ(lhs[i], rhs[i]);
  }
}

void _SimpleTestTwoWorkers(int rank, int world_size) {
  SampleOption options[2];
  {
    int r = 0;
    auto& option = options[r];
    int n_nodes = 4;
    DLContext gpu_context({kDLGPU, r});
    auto src = IdArray::FromVector(std::vector<int64_t>({0, 1, 1})).CopyTo(gpu_context);
    auto dst = IdArray::FromVector(std::vector<int64_t>({1, 2, 3})).CopyTo(gpu_context);
    option.hg = HeteroGraphRef(UnitGraph::CreateFromCOO(1, n_nodes, n_nodes, src, dst));
    option.min_vids = IdArray::FromVector(std::vector<int64_t>({0, 2, 4})).CopyTo(gpu_context);
    option.global_nid_map = IdArray::FromVector(std::vector<int64_t>({0, 1, 2, 3})).CopyTo(gpu_context);
    option.seeds = IdArray::FromVector(std::vector<int64_t>({3, 0})).CopyTo(gpu_context);
    option.n_local_nodes = 2;
    option.n_global_nodes = 4;
    option.fanout = 2;
  }
  {
    int r = 1;
    auto& option = options[r];
    int n_nodes = 3;
    DLContext gpu_context({kDLGPU, r});
    auto src = IdArray::FromVector(std::vector<int64_t>({0, 0, 1})).CopyTo(gpu_context);
    auto dst = IdArray::FromVector(std::vector<int64_t>({1, 2, 0})).CopyTo(gpu_context);
    option.hg = HeteroGraphRef(UnitGraph::CreateFromCOO(1, n_nodes, n_nodes, src, dst));
    option.min_vids = IdArray::FromVector(std::vector<int64_t>({0, 2, 4})).CopyTo(gpu_context);
    option.global_nid_map = IdArray::FromVector(std::vector<int64_t>({2, 3, 0})).CopyTo(gpu_context);
    option.seeds = IdArray::FromVector(std::vector<int64_t>({1, 2})).CopyTo(gpu_context);
    option.n_local_nodes = 2;
    option.n_global_nodes = 4;
    option.fanout = 2;
  }
  auto frontier = SampleNeighbors(options[rank]);
  auto vec = frontier.ToVector<int64_t>();
  if(rank == 0) {
    ExpVectorEq(vec, {2, 2, 1, 1});
  } else {
    ExpVectorEq(vec, {2, 3, 3, 0});
  }
}

TEST(Sampling, Functional) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  _SimpleTestTwoWorkers(rank, world_size);
}

