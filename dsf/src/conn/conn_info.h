/*!
 *  Copyright (c) 2022 by Contributors
 */
#ifndef DGL_DSF_CONN_INFO_H_
#define DGL_DSF_CONN_INFO_H_

#include <cstdint>

#define CACHE_LINE_SIZE 128
#define MEM_ALIGN 4096

namespace dgl {
namespace dsf {

struct ConnMem {
  union {
    struct {
      uint64_t done;
      char pad1[CACHE_LINE_SIZE - sizeof(uint64_t)];
    };
    char pad2[MEM_ALIGN];
  };
  union {
    struct {
      uint64_t ready;
      char pad3[CACHE_LINE_SIZE - sizeof(uint64_t)];
    };
    char pad4[MEM_ALIGN];
  };
  char buffer[1];  // Actually larger than that
};

struct VarArray {
  int64_t *size;
  void *data;
};

struct ConnInfo {
  uint64_t *my_ready, *my_done;
  uint64_t *peer_ready, *peer_done;
  VarArray my_buffer[2], peer_buffer[2];
};

static VarArray BuildVarArray(void *ptr) {
  VarArray ret;
  ret.size = (int64_t *)(ptr);
  ret.data = ptr + MEM_ALIGN;
  return ret;
}

}  // namespace dsf
}  // namespace dgl

#endif  // DGL_DSF_CONN_INFO_H_