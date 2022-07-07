
from ..dataloading import NeighborSampler
from ..transforms import to_block
from .neighbor import sample_neighbors
from ..base import NID


class DistNeighborSampler(NeighborSampler):
    def __init__(self, fanouts):
        super(DistNeighborSampler, self).__init__(fanouts)

    '''
    suppose g, seed_nodes are all on gpu
    '''
    def sample_blocks(self, dataset, seeds, exclude_eids=None):
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            subg = sample_neighbors(dataset, seeds, fanout)

            # Then we compact the frontier into a bipartite graph for message passing.
            block = to_block(subg, seeds)

            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[NID]
            blocks.insert(0, block)
        input_nodes = block.srcdata[NID]
        output_nodes = block.dstdata[NID]
        return input_nodes, output_nodes, blocks