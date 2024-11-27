#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load intel
icx -g -Wall -o ../hw5-OMP ../hw5-OMP.c -fopenmp

.././hw5-OMP 5000 5000 20 /scratch/ualclsd0190/hw5/OMP/5000-5000a.txt
.././hw5-OMP 5000 5000 20 /scratch/ualclsd0190/hw5/OMP/5000-5000b.txt
.././hw5-OMP 5000 5000 20 /scratch/ualclsd0190/hw5/OMP/5000-5000c.txt
.././hw5-OMP 10000 5000 20 /scratch/ualclsd0190/hw5/OMP/10000-5000a.txt
.././hw5-OMP 10000 5000 20 /scratch/ualclsd0190/hw5/OMP/10000-5000b.txt
.././hw5-OMP 10000 5000 20 /scratch/ualclsd0190/hw5/OMP/10000-5000c.txt