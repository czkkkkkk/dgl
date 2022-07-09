import torch as th
import contextlib

from ..transforms.functional import add_self_loop
from .utils import csr_to_global_id, rebalance_train_nids
from ..distributed import load_partition
from ..base import NID
from .. import add_self_loop


class DistributedDataset(object):
    def __init__(self, part_config, batch_size, rank):
        self.rank = rank
        self.batch_size = batch_size
        self.dgl_graph, self.node_feats, _, self.gpb, _, _, _ = load_partition(
            part_config, rank)
        assert '_N/train_mask' in self.node_feats
        self.n_local_nodes = self.node_feats['_N/train_mask'].shape[0]
        self.n_global_nodes = int(self.gpb._max_node_ids[-1])
        self.train_nid = th.masked_select(
            self.dgl_graph.nodes()[:self.n_local_nodes], self.node_feats['_N/train_mask'].bool())
        self.global_nid_map = self.dgl_graph.ndata[NID]

        self._preprocess()

        print('Rank {} loaded graph with {} local nodes, {} edges and {} train nodes'.format(
            rank, self.n_local_nodes, self.dgl_graph.number_of_edges(), self.train_nid.shape[0]))

    def _preprocess(self):
        self.dgl_graph = add_self_loop(self.dgl_graph)
        self.dgl_graph = self.dgl_graph.formats(['csr'])
        self.device = th.device('cuda:%d' % self.rank)
        self.train_nid = rebalance_train_nids(
            self.train_nid, self.batch_size, self.global_nid_map).to(self.device)
        # self.dgl_graph = csr_to_global_id(self.dgl_graph, self.global_nid_map)
        self.dgl_graph = self.dgl_graph.to(self.device)
        self.global_nid_map = self.global_nid_map.to(self.device)
        self.min_vids = [0] + list(self.gpb._max_node_ids)
        self.min_vids = th.tensor(
            self.min_vids, dtype=th.int64).to(self.device)

    # To pass checks in DGL node dataloader
    def get_node_storage(self):
        raise NotImplementedError 

    def get_edge_storage(self):
        raise NotImplementedError 

    def local_scope(self):
        return contextlib.nullcontext()

    @property
    def is_block(self):
        return False
