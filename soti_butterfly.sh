#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=16G
module load scipy-stack
python soti_butterfly.py