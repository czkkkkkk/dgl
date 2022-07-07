from .._ffi.function import _init_api
from .. import backend as F
from ..heterograph import DGLHeteroGraph

'''
def sample_neighbors(g, n_global_nodes, n_local_nodes, min_vids, seeds, global_nid_map, fanout):
    min_vids = F.to_dgl_nd(min_vids)
    seeds = F.to_dgl_nd(seeds)
    global_nid_map = F.to_dgl_nd(global_nid_map)
    ret = _CAPI_DGLDSFSampleNeighbors(g._graph, n_global_nodes, n_local_nodes, min_vids, seeds, global_nid_map, fanout)
    return F.from_dgl_nd(ret)
'''

def sample_neighbors(dataset, seeds, fanout):
    min_vids = F.to_dgl_nd(dataset.min_vids)
    seeds = F.to_dgl_nd(seeds)
    global_nid_map = F.to_dgl_nd(dataset.global_nid_map)
    subgidx = _CAPI_DGLDSFSampleNeighbors(dataset.dgl_graph._graph, dataset.n_global_nodes, dataset.n_local_nodes, min_vids, seeds, global_nid_map, fanout)
    subg = DGLHeteroGraph(subgidx)
    return subg

_init_api("dgl.dsf.neighbor")