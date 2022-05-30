from .._ffi.function import _init_api
from .. import backend as F

__all__ = ['rebalance_train_nids',
           'csr_to_global_id']


def csr_to_global_id(g, global_nid_map):
    global_nid_map = F.to_dgl_nd(global_nid_map)
    _CAPI_DGLDSFCSRToGlobalId(g._graph, global_nid_map)
    return g


def rebalance_train_nids(train_nids, batch_size, global_nid_map, mode="random"):
    train_nids = F.to_dgl_nd(train_nids)
    global_nid_map = F.to_dgl_nd(global_nid_map)
    ret = _CAPI_DGLDSFRebalanceNIds(
        train_nids, batch_size, global_nid_map, mode)
    ret = F.from_dgl_nd(ret)
    return ret


_init_api("dgl.dsf.utils")
