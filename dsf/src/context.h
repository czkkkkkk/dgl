/*!
 *  Copyright (c) 2022 by Contributors
 */
#ifndef DGL_DSF_CONTEXT_H_
#define DGL_DSF_CONTEXT_H_

#include <memory>
#include <string>

#include "./coordinator.h"
#include "./conn/communicator.h"

namespace dgl {
namespace dsf {

struct Context {
  int rank, world_size;
  std::unique_ptr<Communicator> communicator;
  std::unique_ptr<Coordinator> coor;

  static Context* Global() {
    static Context inst;
    return &inst;
  }
};

}  // namespace dsf
}  // namespace dgl

#endif  // DGL_DSF_CONTEXT_H_
