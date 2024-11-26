#!/bin/bash
# load the cuda environment
source /apps/profiles/modules_asax.sh.dyn
module load cuda/11.7.0
nvcc shared.cu -o shared
# run your software here
./shared 20 10 test.txt
