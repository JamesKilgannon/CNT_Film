#!/bin/sh
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --job-name=CNT_network
#SBATCH -p ser-par-10g-3
#SBATCH -e error_%a.log
#SBATCH -o output_%a.log
stdbuf -o0 -e0 python3 -u CNT_Discovery_Model_test.py
