#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load intel
icx -g -Wall -o ../hw5-serial ../hw5-serial.c

.././hw5-serial 5000 5000 /scratch/ualclsd0190/hw5/serial/5000-5000a.txt
.././hw5-serial 5000 5000 /scratch/ualclsd0190/hw5/serial/5000-5000b.txt
.././hw5-serial 5000 5000 /scratch/ualclsd0190/hw5/serial/5000-5000c.txt
.././hw5-serial 10000 5000 /scratch/ualclsd0190/hw5/serial/10000-5000a.txt
.././hw5-serial 10000 5000 /scratch/ualclsd0190/hw5/serial/10000-5000b.txt
.././hw5-serial 10000 5000 /scratch/ualclsd0190/hw5/serial/10000-5000c.txt