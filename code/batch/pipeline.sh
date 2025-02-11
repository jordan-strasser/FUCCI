#!/bin/bash
#SBATCH -t 12:00:00  # Runtime limit (D-HH:MM:SS)
#SBATCH -N 1   # Number of nodes
#SBATCH -n 1   # Number of tasks per node
#SBATCH -c 24   # Number of CPU cores per task
#SBATCH --job-name=fucci   # Job name
#SBATCH --mail-type=FAIL,BEGIN,END     # Send an email when job fails, begins, and finishes
#SBATCH --mail-user=Jordan.Strasser@tufts.edu       # Email address for notifications
#SBATCH --error=%x-%J-%u.err   # Standard error file: <job_name>-<job_id>-<username>.err
#SBATCH --output=%x-%J-%u.out  # Standard output file: <job_name>-<job_id>-<username>.out
/cluster/tufts/levinlab/jstras02/fucci/code/pipeline.py /cluster/home/jstras02/levinlab_link/data/FUCCI/P3

