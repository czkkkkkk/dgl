n_gpus=$1

export DGL_DSF_N_BLOCKS=64
export DGL_DSF_BLOCK_SIZE=512
reddit=/efs/zkcai/projects/dsdgl/examples/pytorch/graphsage/ds/data/reddit${n_gpus}/reddit.json
python test.py --world_size ${n_gpus} \
  --part_config=$reddit \
  --batch_size=1024
