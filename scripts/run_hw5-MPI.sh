#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load openmpi/4.1.4-gcc11

mpicc -g -std=c99 -o ../hw5-MPI ../hw5-MPI.c

# run your software here
mpirun -np 20 .././hw5-MPI 5000 5000 /scratch/ualclsd0190/hw5/MPI/5000-5000b.txt
mpirun -np 20 .././hw5-MPI 5000 5000 /scratch/ualclsd0190/hw5/MPI/5000-5000a.txt
mpirun -np 20 .././hw5-MPI 5000 5000 /scratch/ualclsd0190/hw5/MPI/5000-5000c.txt


mpirun -np 20 .././hw5-MPI 10000 5000 /scratch/ualclsd0190/hw5/MPI/10000-5000a.txt
mpirun -np 20 .././hw5-MPI 10000 5000 /scratch/ualclsd0190/hw5/MPI/10000-5000b.txt
mpirun -np 20 .././hw5-MPI 10000 5000 /scratch/ualclsd0190/hw5/MPI/10000-5000c.txt


