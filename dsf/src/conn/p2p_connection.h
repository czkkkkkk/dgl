/*!
 *  Copyright (c) 2022 by Contributors
 */
#ifndef DGL_DSF_CONN_P2P_CONNECTION_H_
#define DGL_DSF_CONN_P2P_CONNECTION_H_

#include "../utils.h"
#include "./connection.h"

#include <cuda_runtime.h>

namespace dgl {
namespace dsf {

struct P2pExchangeConnInfo {
  cudaIpcMemHandle_t dev_ipc;
  void* ptr;
};

class P2pConnection : public Connection {
 public:
  P2pConnection(ProcInfo my_info, ProcInfo peer_info) :Connection(my_info, peer_info) {}

  void Setup(ConnMem** conn_mem, int buffer_size, ConnInfo* conn_info,
             ExchangeConnInfo* ex_info) override {
    int mem_size = offsetof(ConnMem, buffer) + buffer_size;
    DSFCudaMalloc((char**)conn_mem, mem_size);
    conn_info->my_ready = &(*conn_mem)->ready;
    conn_info->my_done = &(*conn_mem)->done;
    conn_info->my_buffer[0] = BuildVarArray(&(*conn_mem)->buffer);
    conn_info->my_buffer[1] =
        BuildVarArray(&(*conn_mem)->buffer + buffer_size / 2);

    P2pExchangeConnInfo p2p_info;
    if (!my_info_.SameDevice(peer_info_)) {
      cudaIpcGetMemHandle(&p2p_info.dev_ipc, (void*)(*conn_mem));
      static_assert(sizeof(P2pExchangeConnInfo) <= sizeof(ExchangeConnInfo),
                    "P2P exchange connection info too large");
    } else {
      p2p_info.ptr = *conn_mem;
    }
    memcpy(ex_info, &p2p_info, sizeof(P2pExchangeConnInfo));
  }

  void Connect(ConnInfo* conn_info, int buffer_size,
               ExchangeConnInfo* peer_ex_info) override {
    P2pExchangeConnInfo* peer_info = (P2pExchangeConnInfo*)peer_ex_info;
    ConnMem* ptr;
    if (!my_info_.SameDevice(peer_info_)) {
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ptr, peer_info->dev_ipc,
                                     cudaIpcMemLazyEnablePeerAccess));
    } else {
      ptr = (ConnMem*)peer_info->ptr;
    }
    conn_info->peer_ready = &ptr->ready;
    conn_info->peer_done = &ptr->done;
    conn_info->peer_buffer[0] = BuildVarArray(&ptr->buffer);
    conn_info->peer_buffer[1] = BuildVarArray(&ptr->buffer + buffer_size / 2);
  }
};

}  // namespace dsf
}  // namespace dgl

#endif  // DGL_DSF_CONN_P2P_CONNECTION_H_
