/*!
 *  Copyright (c) 2022 by Contributors
 */
#ifndef DGL_DSF_CONN_CONNECTION_H_
#define DGL_DSF_CONN_CONNECTION_H_

#include <memory>

#include "../coordinator.h"
#include "./conn_info.h"

#define MAX_EXCHANGE_CONN_INFO_SIZE 256

namespace dgl {
namespace dsf {

enum ConnType { P2P };

struct ExchangeConnInfo {
  char info[MAX_EXCHANGE_CONN_INFO_SIZE];
};

class Connection {
 public:
  Connection(ProcInfo my_info, ProcInfo peer_info)
      : my_info_(my_info), peer_info_(peer_info) {}

  // Build a connection
  static std::shared_ptr<Connection> BuildConnection(ProcInfo r1, ProcInfo r2);

  virtual void Setup(ConnMem** conn_mem, int buffer_size, ConnInfo* conn_info,
                     ExchangeConnInfo* ex_info) = 0;
  virtual void Connect(ConnInfo* conn_info, int buffer_size,
                       ExchangeConnInfo* peer_ex_info) = 0;

 protected:
  ProcInfo my_info_, peer_info_;
};

}  // namespace dsf
}  // namespace dgl

#endif  // DGL_DSF_CONN_CONNECTION_H_
