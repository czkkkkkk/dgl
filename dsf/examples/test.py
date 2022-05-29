import dgl
import dgl.dsf as dsf
import argparse
import torch.multiprocessing as mp


def run(rank, args):
    world_size = args.world_size
    dsf.init(rank, world_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--world_size', default=4,
                        type=int, help='World size')
    args = parser.parse_args()

    mp.spawn(run, args=(args, ),
             nprocs=args.world_size,
             join=True)
