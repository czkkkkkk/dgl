import torch as th

from python.dgl.transforms.functional import add_self_loop

from .utils import csr_to_global_id, rebalance_train_nids
from ..distributed import load_partition
from .. import NID
from .. import add_self_loop


class DistributedDataset(object):
    def __init__(self, part_config, batch_size, rank):
        self.rank = rank
        self.batch_size = batch_size
        self.g, self.node_feats, _, self.gpb, _, _, _ = load_partition(
            part_config, rank)
        assert '_N/train_mask' in self.node_feats
        self.n_local_nodes = self.node_feats['_N/train_mask'].shape[0]
        self.n_global_nodes = self.gpb._max_node_ids[-1]
        self.train_nid = th.masked_select(
            self.g.nodes()[:self.n_local_nodes], self.node_feats['_N/train_mask'].bool())
        self.global_nid_map = self.g.ndata[NID]

        self._preprocess()

        print('Rank {} loaded graph with {} local nodes, {} edges and {} train nodes'.format(
            rank, self.n_local_nodes, self.g.number_of_edges(), self.train_nid.shape[0]))

    def _preprocess(self):
        self.g = add_self_loop(self.g)
        self.g = self.g.formats(['csr'])
        self.device = th.device('cuda:%d' % self.rank)
        self.train_nid = rebalance_train_nids(
            self.train_nid, self.batch_size, self.global_nid_map).to(self.device)
        self.g = csr_to_global_id(self.g, self.global_nid_map)
        self.global_nid_map = self.global_nid_map.to(self.device)
        self.min_vids = [0] + list(self.gpb._max_node_ids)
        self.min_vids = th.tensor(
            self.min_vids, dtype=th.int64).to(self.device)
