#!/bin/bash
# load the cuda environment
source /apps/profiles/modules_asax.sh.dyn
module load cuda/11.7.0
nvcc hello.cu
# run your software here
./a.out