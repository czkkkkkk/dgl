import dgl
import torch as th
import argparse
import torch.multiprocessing as mp
import dgl.dsf as dsf
import time


def run(rank, args):
    world_size = args.world_size
    dsf.init(rank, world_size)
    dataset = dsf.DistributedDataset(
        args.part_config, args.batch_size, rank)
    sampler = dsf.sampler.DistNeighborSampler([10, 25])
    dataloader = dgl.dataloading.NodeDataLoader(
        dataset,
        dataset.train_nid,
        sampler,
        device=dataset.device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    for i in range(10):
        start = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # print('Rank {}, input_nodes, output_nodes, sampled blocks 0 {}'.format(rank, input_nodes, seeds, blocks[0]))
            pass
        end = time.time()
        print("Rank {} using time for an epoch {}".format(rank, end - start))
    


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