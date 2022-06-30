/*!
 *  Copyright (c) 2022 by Contributors
 */

#ifndef DGL_DSF_NEIGHBOR_H_
#define DGL_DSF_NEIGHBOR_H_

#include <dgl/array.h>

#include "graph/unit_graph.h"

namespace dgl {
namespace dsf {

struct SampleOption {
  HeteroGraphRef hg;
  IdArray seeds; // Seeds with global node ids
  IdArray global_nid_map;
  IdArray min_vids;
  int64_t fanout;
  int64_t n_local_nodes, n_global_nodes;
};

HeteroGraphPtr SampleNeighbors(const SampleOption& option);

}  // namespace dsf
}  // namespace dgl

#endif // DGL_DSF_NEIGHBOR_H_
