#!/bin/bash
#PBS -l nodes=1:ppn=64
#PBS -N unoptimization
#PBS -o /home/yusei/mcr/cluster/pbso
#PBS -e /home/yusei/mcr/cluster/pbse
ulimit -s unlimited
cd ${PBS_O_WORKDIR}

python mcr_benchmark.py $p