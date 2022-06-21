n_gpus=4

reddit=/efs/zkcai/projects/dsdgl/examples/pytorch/graphsage/ds/data/reddit${n_gpus}/reddit.json
python test.py --world_size ${n_gpus} \
  --part_config=$reddit
