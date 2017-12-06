#!/bin/bash
sbatch --exclude=holygpu[01-16],holyseasgpu[01-13],shakgpu[01-50],aaggpu[01-08],supermicgpu01 slurm_script.sh > job_id.txt &
