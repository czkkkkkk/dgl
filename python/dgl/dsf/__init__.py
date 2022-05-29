from .._ffi.function import _init_api



def init(rank, world_size):
    _CAPI_DGLDSFInitialize(rank, world_size)

_init_api("dgl.dsf")