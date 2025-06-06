#!/bin/bash
#PBS -l nodes=1:ppn=64
#PBS -N unoptimization
#PBS -o /home/yusei/mcr/pbso
#PBS -e /home/yusei/mcr/pbse
ulimit -s unlimited
cd ${PBS_O_WORKDIR}

source .venv/bin/activate
python exp_mcr_benchmark.py $p