#!/bin/bash
# load the cuda environment
source /apps/profiles/modules_asax.sh.dyn
module load cuda/11.7.0
nvcc ../hw5-GPU.cu -o ../hw5-GPU
# run your software here
.././hw5-GPU 5000 5000 /scratch/ualclsd0190/hw5/GPU/5000-5000a.txt
.././hw5-GPU 5000 5000 /scratch/ualclsd0190/hw5/GPU/5000-5000b.txt
.././hw5-GPU 5000 5000 /scratch/ualclsd0190/hw5/GPU/5000-5000c.txt


.././hw5-GPU 10000 5000 /scratch/ualclsd0190/hw5/GPU/10000-5000a.txt
.././hw5-GPU 10000 5000 /scratch/ualclsd0190/hw5/GPU/10000-5000b.txt
.././hw5-GPU 10000 5000 /scratch/ualclsd0190/hw5/GPU/10000-5000c.txt



