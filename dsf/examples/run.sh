n_gpus=$1

export DGL_DSF_N_BLOCKS=64
export DGL_DSF_BLOCK_SIZE=512
prod=/efs/zkcai/projects/dsdgl/examples/pytorch/graphsage/ds/data/ogb-product${n_gpus}/ogb-product.json
reddit=/efs/zkcai/projects/dsdgl/examples/pytorch/graphsage/ds/data/reddit${n_gpus}/reddit.json
dataset=$reddit
dataset=$prod

python test.py --world_size ${n_gpus} \
  --part_config=$dataset \
  --batch_size=1024
