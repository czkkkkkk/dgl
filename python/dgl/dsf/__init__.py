from .._ffi.function import _init_api
from .dataset import DistributedDataset


def init(rank, world_size):
    _CAPI_DGLDSFInitialize(rank, world_size)


_init_api("dgl.dsf")
