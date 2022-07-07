/*!
 *  Copyright (c) 2022 by Contributors
 */
#ifndef DGL_DSF_CONN_COMMUNICATOR_H_
#define DGL_DSF_CONN_COMMUNICATOR_H_

#define MAX_NUM_COMM_BLOCKS 64

#include "./conn_info.h"
#include "./connection.h"

namespace dgl {
namespace dsf {

struct BlockConn {
  ConnMem** conn_mems;
  ConnInfo* conn_infos;
};

struct Communicator {
  int n_blocks;
  BlockConn block_conns[MAX_NUM_COMM_BLOCKS];
  Communicator* dev_communicator;
};

void SetupCommunicator(Communicator* communicator);

}  // namespace dsf
}  // namespace dgl

#endif  // DGL_DSF_CONN_COMMUNICATOR_H_
