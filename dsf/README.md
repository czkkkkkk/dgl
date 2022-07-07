# Distributed Sampling with Kernel Fusion

DSF conducts graph sampling on multiple GPUs collectively with kernel optimizations. In DSF, we use pytorch DDP (distribted data parallel) package to spawn multiple processes running on the same code. Each process is mapped to exactly one GPU and loads a partitioned subgraph from disk. The graph structure is stored on GPUs. DSF adopts a task-exchange paradigm to conduct sampling collectively, where GPUs exhange their sampling tasks if the nodes to be sampled is partitioned to other GPUs. DSF also uses aggresive kernel optimizations to generate high-performance cuda code to execute the sampling on multiple GPUs.


## Installation

DSF is extended from DGL to reuse some key components in DGL such as the ffi, DGLGraph and NDArray modules. Similarly to DGL, installing DSF needs to compile the C++/CUDA code into a shared object (so) and then bind the so with python interfaces. 

### Building C++/CUDA code

```sh
cd build
# Also set(USE_CUDA ON) in the config
cp ../cmake/config.cmake .
# Add a -DCMAKE_CXX_COMPILER_LAUNCHER=ccache flag to acclerate the building process
cmake -DUSE_NCCL=ON -DCMAKE_BUILD_TYPE=Release ..
make -j
```

### Install python package
It is highly recommended to install DSF on a seperated environment like conda.

```sh
cd ../python
python setup.py install
```

## Running test

The DSF unittests are in the directory `dsf/tests`. The tests are automatically built and does not affected by the `-DBUILD_CPP_TEST` in DGL. The test environment is run on MPI, which can be run with:
```sh
cd build
mpirun -np 2 ./runUnitTests
```

Note that currently our tests assume sequential sampling instead of random sampling. To pass the tests, we need to use the sequential sampling, which uncomments about the 125 line in `dsf/src/cuda/sampling_kernel.cu`.

## Code Structure

### C++/CUDA
`dsf/src` is the directory of C++/CUDA code, which contains the communication context between GPUs and the communication kernel.

### Python
`python/dgl/dsf` is the python code of dsf, which contains the distributed graph representation and sampling interface.

## Running an example

```sh
cd dsf/examples
bash example_run.sh 2
```
This example runs a graphsage on the reddit or ogbn-products datasets. In the example, we use DistributedDataset to load the partitioned graph from disk and define a DistNeighborSampler for the DGL NodeDataLoader to conduct sampling.

## Key Components
TODO...
### DistNeighborSampler
### DistributedDataset
### Coordinator
### Communicator
### GPU Connection

## Linting
```sh
cd dsf
bash ./scripts/lint.sh
```

This script will check the cpp code in DSF and print the code that violates the format rules. To make the formatting easier, it is suggested to format the code in the IDE using `dsf/.clang-format` first.