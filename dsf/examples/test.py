import dgl
import dgl.dsf as dsf
import argparse
import torch.multiprocessing as mp


def run(rank, args):
    world_size = args.world_size
    dsf.init(rank, world_size)
    dataset = dsf.DistributedDataset(
        args.part_config, args.batch_size, rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--world_size', default=4,
                        type=int, help='World size')
    parser.add_argument('--part_config', default='',
                        type=str, help='The path to the partition config file')
    parser.add_argument('--batch_size', default=1024,
                        type=int, help='Batch size')
    args = parser.parse_args()

    mp.spawn(run, args=(args, ),
             nprocs=args.world_size,
             join=True)
