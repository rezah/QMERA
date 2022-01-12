
#watch -n 2 nvidia-smi
#watch -n 3 nvidia-smi --query-gpu=index,gpu_name,memory.total,memory.used,memory.free,temperature.gpu,pstate,utilization.gpu,utilization.memory --format=csv
#CUDA_VISIBLE_DEVICES=3


#conda
conda env list
conda activate


#parallel-ray
ray stop --force
ray start --head --num-cpus=12 --temp-dir=~/Dropbox/Prog/MERA/temp/


#parallel
taskset --cpu-list 0-3 python3 -W ignore energy-quimb.py
ulimit -n 50000
htop
sar 4      #every 4 seconds




#slurm
seff  18779642
sacct -j18714378 --format=JobID,Start,End,Elapsed,NCPUS
#node info
scontrol  show node  hpc-81-33
sinfo --Node --long
#user
ulimit -a
sacct -u haqshena


#slurm_memory
my_home_usage



#Slurm
Yes there is a difference between those two submissions. You are correct that usually ntasks is for mpi and cpus-per-task is for multithreading, but let’s look at your commands:

For your first example, the sbatch --ntasks 24 […] will allocate a job with 24 tasks. These tasks in this case are only 1 CPUs, but may be split across multiple nodes. So you get a total of 24 CPUs across multiple nodes.

For your second example, the sbatch --ntasks 1 --cpus-per-task 24 [...] will allocate a job with 1 task and 24 CPUs for that task. Thus you will get a total of 24 CPUs on a single node.

In other words, a task cannot be split across multiple nodes. Therefore, using --cpus-per-task will ensure it gets allocated to the same node, while using --ntasks can and may allocate it to multiple nodes.




you use mpi and do not care about where those cores are distributed: --ntasks=16
you want to launch 16 independent processes (no communication): --ntasks=16
you want those cores to spread across distinct nodes: --ntasks=16 and --ntasks-per-node=1 or --ntasks=16 and --nodes=16
you want those cores to spread across distinct nodes and no interference from other jobs: --ntasks=16 --nodes=16 --exclusive
you want 16 processes to spread across 8 nodes to have two processes per node: --ntasks=16 --ntasks-per-node=2
you want 16 processes to stay on the same node: --ntasks=16 --ntasks-per-node=16
you want one process that can use 16 cores for multithreading: --ntasks=1 --cpus-per-task=16
you want 4 processes that can use 4 cores each for multithreading: --ntasks=4 --cpus-per-task=4






