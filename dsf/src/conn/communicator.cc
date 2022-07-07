/*!
 *  Copyright (c) 2022 by Contributors
 */
#include "./communicator.h"
#include "../context.h"
#include "../utils.h"

namespace dgl {
namespace dsf {

void BuildBlockConnInfo(const std::vector<std::shared_ptr<Connection>>& conns,
                        Coordinator* coor, BlockConn* block_conn) {
  int rank = coor->GetRank();
  int world_size = coor->GetWorldSize();
  std::vector<ConnMem*> conn_mems(world_size);
  std::vector<ConnInfo> conn_infos(world_size);
  static constexpr int BUFFER_SIZE = 4 * 1024 * 1024;
  for (int offset = 0; offset < world_size; ++offset) {
    int next = (rank + offset) % world_size;
    int prev = (rank + world_size - offset) % world_size;
    ExchangeConnInfo ex_info;
    conns[prev]->Setup(&conn_mems[prev], BUFFER_SIZE, &conn_infos[prev],
                       &ex_info);
    auto next_ex_info = coor->RingExchange(next, ex_info);
    conns[next]->Connect(&conn_infos[next], BUFFER_SIZE, &next_ex_info);
  }
  DSFCUDAMallocAndCopy(&block_conn->conn_mems, conn_mems);
  DSFCUDAMallocAndCopy(&block_conn->conn_infos, conn_infos);
}

void BuildCommunicator(int n_blocks,
                       const std::vector<std::shared_ptr<Connection>>& conns,
                       Coordinator* coor, Communicator* communicator) {
  communicator->n_blocks = n_blocks;
  for (int bid = 0; bid < n_blocks; ++bid) {
    BuildBlockConnInfo(conns, coor, &communicator->block_conns[bid]);
  }
  DSFCUDAMallocAndCopy(&communicator->dev_communicator, communicator, 1);
}

void SetupCommunicator(Communicator* communicator) {
  auto* context = Context::Global();
  int world_size = context->world_size;
  int rank = context->rank;
  auto* coor = context->coor.get();
  std::vector<std::shared_ptr<Connection>> conns;
  for (int r = 0; r < world_size; ++r) {
    conns.push_back(Connection::BuildConnection(coor->GetPeerInfos()[rank],
                                                coor->GetPeerInfos()[r]));
  }
  int n_blocks = GetEnvParam("N_BLOCKS", 16);
  BuildCommunicator(n_blocks, conns, coor, communicator);
}

}  // namespace dsf
}  // namespace dgl
