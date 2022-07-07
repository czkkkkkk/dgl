/*!
 *  Copyright (c) 2022 by Contributors
 */
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "core.h"
#include "utils.h"

using namespace dgl::dsf;

GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  LOG(INFO) << "MPI test is enabled";
  MPI_Init(&argc, &argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  SetEnvParam("MASTER_PORT", 12307);
  Initialize(rank, world_size);

  int result = RUN_ALL_TESTS();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return result;
}
