
import dgl


class NeighborSampler(object):
    def __init__(self, g, fanouts, sample_neighbors, device, load_feat=True):
        # g is a DistributedDataset
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.load_feat = load_feat
        self.device = device

    '''
    suppose g, seed_nodes are all on gpu
    '''

    def sample_blocks(self, g, seeds, exclude_eids=None):
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.g, fanout)

            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)

            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks
