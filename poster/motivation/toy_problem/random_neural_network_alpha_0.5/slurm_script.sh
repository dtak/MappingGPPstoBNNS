#!/bin/bash

#SBATCH -p serial_requeue # Partition to submit to (comma separated)
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 2-00:00 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem 8000 # Memory in MB (see also --mem-per-cpu)
#SBATCH -o output_process # File to which standard out will be written
#SBATCH -e errors_process # File to which standard err will be written
#SBATCH --gres=gpu:1

#module load hpc/mongodb

random_seed=`cat /dev/urandom | tr -cd 'a-f0-9' | head -c 32`
COMPILE_DIR=`echo /scratch/theano_compile_dir_`$random_seed 
echo $COMPILE_DIR
mkdir $COMPILE_DIR
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,gcc.cxxflags="-march=corei7",base_compiledir=$COMPILE_DIR ~/bin/python experiment.py
