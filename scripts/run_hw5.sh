#!/bin/bash
# load the cuda environment
source /apps/profiles/modules_asax.sh.dyn
module load cuda/11.7.0
nvcc ../hw5.cu -o ../hw5
# run your software here
.././hw5 100 100 out.txt