 
============================================================
=====     Queued job information at submit time        =====
============================================================
  The submitted file is: run_hw5-GPU.sh
  The time limit is 01:00:00 HH:MM:SS.
  The target directory is: /home/ualclsd0190/Fall2024_CS481_HW5/scripts
  The memory limit is: 20gb
  The job will start running after: 202411261817.25
  Job Name: runhw5GPUshGPU
  Queue: -q classgpu
  Constraints: 
  Command typed:
/scripts/run_gpu run_hw5-GPU.sh     
  Queue submit command:
qsub -q classgpu -j oe -N runhw5GPUshGPU -a 202411261817.25 -r n -M cgphornroekngam@crimson.ua.edu -l walltime=01:00:00 -l select=1:ngpus=1:ncpus=1:mpiprocs=1:mem=20000mb 
  Job number: 
 
