#!/bin/bash
# load the cuda environment
source /apps/profiles/modules_asax.sh.dyn
module load cuda/11.7.0

nvcc hw5-shared.cu -o hw5-shared

./hw5-shared 1000 5000 /home/ualclsd0190/Fall2024_CS481_HW5/shared-memory/a.txt
./hw5-shared 1000 5000 /home/ualclsd0190/Fall2024_CS481_HW5/shared-memory/b.txt

diff /home/ualclsd0190/Fall2024_CS481_HW5/shared-memory/a.txt /home/ualclsd0190/Fall2024_CS481_HW5/shared-memory/b.txt